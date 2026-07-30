# Stage 3 report

## Decision

**`GO_TO_STAGE_4`**

All twelve frozen Stage 3 gates passed. The generated champion cleared the
development-set threshold against random, retained the required fraction of
the reviewed structural baseline, replayed exactly, and completed without an
invalid graph, worker failure, timeout, crash, protocol failure, oracle call,
or runtime network call.

This decision closes only the Stage 3 development gate. Stage 4 has not
started and requires a separate issue and explicit user approval. The result
is non-held-out development evidence. It is not a held-out generalization
claim, an HEG superiority claim, or authorization to integrate a policy into
HEG.

## Dependency and repository provenance

- Required Mutation Forge entry point:
  `1670f7b023dcf110259ea39b63ba1a55cb011521`.
- Stage 2D issue #8 was closed as completed with `GO_TO_STAGE_3`.
- HEG remained read-only, clean, and pinned to
  `fd97451b0f3d87400d1d955a2c6b1b18303344ff`.
- Dedicated branch: `agent/stage3-issue-9-rebuild`.
- Generation-source freeze commit:
  `413e9fe1a86311c45083571655546089e878ef21`.
- Retained generation-source tag: `stage3-generation-frozen-v11`.
- Provider-free validator amendment tag:
  `stage3-generation-frozen-v12`.
- Final evaluation-only freeze commit:
  `b28b5be350232b94961a21248d9da77814f8da74`.
- Final annotated tag: `stage3-generation-frozen-v13`.
- Stage 3 development-manifest input SHA-256 recorded by the frozen config:
  `7d7cf3cb1cccaea57bbc5ef168845c82ac7be59da7ad8a9613c77bffaa9573f1`.
- Stage 3 canonical development-manifest SHA-256:
  `f94757d7b21b28a16fb1d3d3b4e54385785f07270da8f89496c1cb24e76c96d1`.

Every freeze was committed, pushed, tagged, and recorded on issue #9 before
the corresponding provider or evaluation boundary was crossed. Earlier tags
and failed-attempt evidence were retained rather than moved or overwritten.

## App Server generation

The product path used the installed Codex App Server directly through the
thin local adapter. It ran the frozen profile:

- model: `gpt-5.6-luna`;
- reasoning effort: `high`;
- eight ordered initial slots;
- exactly eight concurrent initial threads and turns;
- at most one schema/AST-only repair turn per slot;
- `approval_policy = "never"`;
- an isolated capsule per slot;
- no repository or benchmark-artifact roots;
- no dynamic tools, shell, filesystem, MCP, plugin, browser, or
  network-research authority.

The campaign made exactly eight initial generation turns and five permitted
repair turns. It made no replacement calls. Exact aggregate usage over all 13
turns was:

| Usage field | Tokens |
|---|---:|
| Input | 61,886 |
| Cached input | 5,376 |
| Output | 43,267 |
| Reasoning output | 36,682 |
| Total | 105,153 |

All eight retained final responses came from those frozen turns. No model or
App Server call occurred during revalidation, evaluation, replay, or report
production.

Each slot retains its rendered prompt, final response, structured result,
JSON-RPC wire transcript, normalized event stream, usage records, stderr, and
transport metadata below:

```text
runs/stage3-development/
  stage3-generation-1f7f0784e37c-attempt-01/
    slots/slot-00/ ... slots/slot-07/
```

The final authority audit parsed every retained JSONL record, matched all
turns to the frozen model and effort, confirmed exact usage, found empty
stderr for every final campaign slot, and found no tool, MCP, plugin, browser,
shell, runtime-network, approval, credential, workspace-root, benchmark
result, baseline result, oracle result, or other-candidate leakage.

## Prompts, schemas, and validation

The schema-derived prompts are self-contained. They state the graph-rewrite
ranking objective, distinguish pool-constant context from proposal-specific
signals, define every field and aligned count vector, disclose contract-level
aliases, and provide the exact Python capability boundary. The versioned
semantic glossary is tied to the frozen context and proposal schema hashes.
All eight rendered prompts are checked-in snapshots.

The App Server transport schema uses only keywords supported by the installed
protocol. The application parser separately enforces exact keys, types,
lengths, cardinalities, and source limits.

The retained v11 campaign originally accepted seven candidates. The remaining
response contained a finite dynamic `range(min(...))` loop that the former
static loop-bound heuristic could not prove. The v12 validator amendment
removed loop-bound and termination inference while preserving the AST
capability boundary and the isolated runtime's CPU and wall limits. This
amendment made no model call and changed no response.

Provider-free revalidation then produced:

- eight accepted candidates;
- eight unique source and normalized-AST identities;
- 10,000 persistent-worker calls per candidate;
- 80,000 successful bounded calls in total;
- zero model or App Server calls.

The frozen generated policies and both reviewed baselines all execute through
the Stage 2A worker. The champion identity is:

| Policy | Source SHA-256 | Normalized AST SHA-256 | AST nodes |
|---|---|---|---:|
| `candidate-slot-04` | `a5f540459695bbf7d454eeccbb8e48158d6130df6a769b67d1447de18276dc01` | `cef05bb644e2e0a9acbc4972fbaa6d4ba3e033ee8a73ecd756da44100c767f5c` | 208 |

## Frozen development experiment

The immutable development manifest contains 128 episodes:

- toy connected-cubic orders 10 and 12;
- graph seeds 301–304;
- policy seeds 3001–3016;
- horizon 32;
- eight generated candidates plus the reviewed random and structural
  baselines;
- the unchanged bounded Stage 2B proposal and feature contract;
- one authoritative score for the initial graph and selected-plan scoring
  only thereafter.

Each policy owns its current graph and score. Accepted strict improvements
advance only that policy's graph, and proposal pools are generated from the
policy's own current graph after trajectories diverge. The Stage 2B
static-source repeated-pool behavior is not used.

The evaluation used eight CPU workers pinned to CPUs 0–7 and reserved CPUs
8–15. Primary and replay recorded the same affinity. All OpenMP, OpenBLAS,
MKL, NumExpr, vecLib, and BLIS thread counts were one.

## Bounded artifact correction

The first v12 evaluation completed primary and replay computation in memory
but failed before durable reduction. It attempted to serialize all 128
episodes into one artifact while duplicating full step traces, score objects,
timing, and resource data across policies. The uncompressed payload crossed
the writer's 64 MiB bound before gzip or a temporary output file was created.
No metric, champion, replay hash, or scientific decision was retained from
that attempt.

The user-authorized v13 correction changed only persistence:

- each shared step trace is stored once;
- redundant complete score objects are replaced by bounded selected-plan
  ordering keys, witness counts, and deltas;
- primary and replay are each split into exactly eight deterministic shards
  of 16 episodes;
- every shard is checked against the 33,554,432-byte uncompressed artifact
  limit;
- replay verifies every shard hash, count, assignment, and canonical aggregate
  hash.

It changed no candidate, baseline, prompt, scientific schema, episode, seed,
metric, bootstrap, threshold, or gate and made no model call.

Observed final artifact sizes:

| Pass | Records | Uncompressed total | Largest shard | Per-shard limit |
|---|---:|---:|---:|---:|
| Primary | 128 | 45,841,430 B | 5,733,843 B | 33,554,432 B |
| Replay | 128 | 45,841,396 B | 5,733,839 B | 33,554,432 B |

## Primary development result

The selected champion was `candidate-slot-04`.

| Policy | Pooled median normalized best-so-far AUC | Median best total witnesses |
|---|---:|---:|
| Champion | 0.929167 | 0 |
| Random | 0.807292 | 0 |
| Structural | 0.932292 | 0 |

Against random:

- paired median AUC delta: `0.098958`;
- relative median improvement: `15.097%`;
- hierarchical paired 95% bootstrap interval:
  `[0.075000, 0.125000]`;
- frozen threshold: at least `5%` relative improvement with the interval
  excluding zero.

Against structural:

- paired median AUC delta: `0.000000`;
- paired 95% bootstrap interval:
  `[-0.005208, 0.008333]`;
- pooled median retention:
  `0.9291666667 / 0.9322916667 = 99.665%`;
- frozen threshold: at least `90%` retention.

Order-specific medians:

| Order | Champion AUC | Random AUC | Structural AUC | Champion best witnesses |
|---:|---:|---:|---:|---:|
| 10 | 0.940625 | 0.828125 | 0.950000 | 0 |
| 12 | 0.916667 | 0.796875 | 0.903646 | 0 |

The bootstrap used 10,000 samples, seed 2026072909, and a 95% percentile
interval.

## Replay, accounting, and safety

Primary and replay each contain 128 timing-stripped episode records. Their
canonical hashes match exactly:

```text
43dee7e356ccc3f11c3fff326a78d16c70b0524a5b046732f6aca289335ccd73
```

The shard-manifest file hashes are:

- primary:
  `5f8fafbb62664f490ef822d5ccf7ba16dd47d68bb46eb3d29c138ccd215637e8`;
- replay:
  `293a9b80e0ad148ff7eaddda7a2af1e0bd0e56fc42140650eb17b04c8bd109ed`.

Primary and replay accounting matched exactly:

- policy-step evaluations: 40,960;
- selected-plan score calls: 40,960;
- initial score calls: 128;
- shared-pool policy steps: 128;
- independent-pool policy steps: 39,680;
- oracle score calls: 0;
- invalid graphs or records: 0;
- policy or worker failures: 0;
- timeouts, crashes, and protocol failures: 0;
- evaluation-time model calls: 0;
- evaluation-time App Server calls: 0;
- experiment-runtime network calls: 0.

## Gate result

| Frozen gate | Result |
|---|---|
| `baseline_ast_distinct` | PASS |
| `campaign_authority` | PASS |
| `champion_random_relative` | PASS |
| `champion_structural_relative` | PASS |
| `dependency_provenance` | PASS |
| `exact_usage` | PASS |
| `minimum_unique` | PASS |
| `primary_replay_exact` | PASS |
| `protocol_safety` | PASS |
| `repository_and_heg_validation` | PASS |
| `selected_only_equal_bounded_parity` | PASS |
| `zero_invalid_and_worker_failures` | PASS |

## Durable artifacts

The complete retained campaign and evaluation evidence is under:

```text
runs/stage3-development/stage3-generation-1f7f0784e37c-attempt-01/
```

Key artifacts:

- `generation_summary.json`, per-slot prompts, responses, usage, events, wire
  logs, and transport logs;
- `revalidation_summary.json`;
- `evaluation-primary-shards.json`;
- `evaluation-primary-shard-00.jsonl.gz` through
  `evaluation-primary-shard-07.jsonl.gz`;
- `evaluation-replay-shards.json`;
- `evaluation-replay-shard-00.jsonl.gz` through
  `evaluation-replay-shard-07.jsonl.gz`;
- `evaluation_summary.json`;
- `gate.json`.

## Validation

Before the final documentation commit:

- full `pytest`: 295 passed, zero skipped;
- Ruff: passed;
- strict mypy: passed for 65 source files;
- `mforge doctor`: passed;
- `git diff --check`: passed;
- HEG: exact pin and clean.

Stage 4, evolution, archive search, held-out evaluation, full proposer work,
and HEG policy integration were not started.
