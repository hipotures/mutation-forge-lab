# Native v3 Step 12D: lineage forks and bounded Search Memory

Step 12D adds a standalone App Server experiment for exact-parent lineage
forks and fresh specification-root forks. It does not change production Native
v3 routing, the scheduler, or Native v2.

## Protocol

The experiment creates one isolated durable source thread:

1. a specification-only anchor with the complete Native v3 program contract;
2. one validated parent-program turn containing the exact canonical AST;
3. one later sibling-program turn;
4. a child fork at the inclusive parent `lastTurnId`;
5. a fresh fork at the inclusive specification-anchor `lastTurnId`.

The child history must contain the specification and exact parent, and must not
contain the later sibling. The child receives compact evaluation feedback,
then generates one mutated direct-program response. If that response fails the
host program contract, the same child fork receives the exact bounded
validation error and gets one repair attempt; no further retry is allowed.

The fresh history must contain only the specification anchor. It receives a
semantic projection of the host-owned `SearchMemoryV1`, then generates one
structurally different fresh root. The host record contains protocol and
program identities, behavior signatures, bounded successful, tested, and
pending pattern summaries, active lineage summaries, validated archive IDs,
and one active parent reference. The model projection contains only short
aliases and semantic summaries: selector and action families, control flow,
contract status, scientific outcome, model hypothesis, observed effect, and
lineage descriptions. It contains neither a full AST nor any cryptographic
identity.

App Server `thread/fork` responses are accepted only when the child is durable,
the source identity matches, the child path remains inside the isolated
capsule, every returned history turn is completed, and the last included turn
is the requested inclusive boundary. Source and child thread IDs, session IDs,
rollout paths, included turn IDs, generations, slots, parent IDs, compact
feedback, program hashes, and behavior signatures are retained in the report.

Host duplicate checks remain authoritative. A generated program is rejected
when either its canonical program hash or behavior signature already exists in
Search Memory; model instructions are advisory.

Step 12C classified explicit compaction as `BEST_EFFORT_ONLY`, so Step 12D does
not use compaction. The fresh specification fork plus bounded Search Memory is
the tested alternative.

## Artifacts

The experiment writes:

- `lineage-report.json.gz` and `lineage-report.md`;
- `child-program.json.gz` and `fresh-root-program.json.gz` for valid results;
- nine provider prefixes when the first child is valid, or ten when the bounded
  child repair is used, including two `thread/fork` RPC prefixes; every prefix
  has the standard 16-file App Server artifact set.

All request Markdown files are exact JSON documents. In particular, the
specification anchor and Search Memory requests can be parsed directly as
JSON. The full host Search Memory canonical bytes and SHA-256 are recorded in
the report but are not copied into the model-facing request.

## Bounded live result

The accepted evidence is
`workspace/step12d_lineage_medium_005/lineage-report.json.gz`. It used
`gpt-5.6-luna` with `medium` reasoning effort.

- Both `thread/fork` boundaries were exact. The child contained the
  specification and exact parent but not the later sibling; the fresh fork
  contained only the specification anchor.
- The first child attempt failed the host terminal-path rule. The one permitted
  repair stayed in the same child thread after compact feedback and produced a
  valid, non-duplicate program.
- The fresh root was valid, non-duplicate, and behaviorally different from the
  four Search Memory entries.
- The accepted run used three successful and one tested pattern summary. Its
  request contained no full program AST or host-owned cryptographic identity.
- The run used eight provider turns and two fork RPCs. All ten executed
  prefixes had exactly the standard 16 artifacts; no rollout copy was added.

| Result | Program hash | Behavior signature |
| --- | --- | --- |
| Repaired lineage child | `34784f81a8dccaffaf6b5951b3608548f7e09c37e819059531fc0046c2710740` | `db459617ad3146c2e710036177ae75521d1b9e85e6ed38638f1735e07d0a1fcd` |
| Fresh root | `85913624d7a8cb0d7c4947501732bbb52c56f39407d5b6dd1730bbfaf3be5ec9` | `c19fa425d7b3b87ea3354e798c46a0360ecc0b1da6799a2476cfac646ce12ada` |

Prompt sizes were 571 bytes for the child request, 536 bytes for its repair,
3,868 bytes for Search Memory, and 375 bytes for the fresh-root request.

| Usage | Tokens |
| --- | ---: |
| Input | 39,997 |
| Cached input | 16,640 |
| Cache-write input | 0 |
| Output | 1,398 |
| Reasoning output | 655 |
| Total | 41,395 |

The disposable workspaces `medium_001` through `medium_004` remain unchanged
as failure evidence. They exposed, in order, delayed fork notifications,
post-fork notification leakage, and two invalid first-attempt child programs.

## Validation

```text
uv run ruff check src/mutation_forge/stage3/app_server.py src/mutation_forge/native_v3/search_memory.py src/mutation_forge/native_v3/lineage_experiment.py scripts/native_v3_lineage_experiment.py tests/fixtures/fake_stage3_app_server.py tests/unit/test_native_v3_lineage_experiment.py
uv run mypy src/mutation_forge/stage3/app_server.py src/mutation_forge/native_v3/search_memory.py src/mutation_forge/native_v3/lineage_experiment.py
uv run pytest -q tests/unit/test_native_v3_lineage_experiment.py
uv run pytest tests/unit/test_native_v3*.py tests/integration/test_native_v3_route.py -q
uv run pytest tests/unit/test_native_v2_smoke.py tests/integration/test_native_experiment.py tests/unit/test_native_resume.py tests/unit/test_native_selection.py tests/unit/test_native_progress.py -q
make appserver-artifact-parity
```

The focused suite covers valid, invalid, and in-progress fork boundaries,
foreign source rejection, delayed post-response notifications, exact sibling
exclusion, deterministic Search Memory ordering and bounds, both duplicate
identities, one bounded child repair, and exact provider artifact suffixes.

## Manual operator command

Run from the dedicated worktree with a new disposable workspace:

```bash
uv run python scripts/native_v3_lineage_experiment.py \
  --workspace workspace/step12d_lineage_medium_001 \
  --auth-json /home/user/.codex/auth.json \
  --model gpt-5.6-luna \
  --effort medium \
  --turn-timeout 900
```

A failed workspace is immutable evidence and must not be reused.

## Scope

This step does not select a production communication mode. Scheduler
integration, automatic lineage selection, production Search Memory injection,
and guarded rollout belong to Step 12E.

STOP — waiting for operator acceptance.
