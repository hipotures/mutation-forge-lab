# Stage 3 report

## Terminal decision

**`INCONCLUSIVE_INFRASTRUCTURE_FAILURE`**

The complete offline Stage 3 implementation was frozen and validated, but the
live campaign could not begin. The required isolated Codex home was not
authenticated, and the installed CLI exposes no supported reference to the
existing authentication store that preserves the issue's isolation boundary.
Mutation Forge failed closed before `thread/start` or `turn/start`.

This is not `NO_GO`: no generated candidate or development result was
observed, so the scientific and yield gates were not evaluated. It is not
`GO_TO_STAGE_4`: Stage 4 remains blocked.

## Frozen provenance

- Mutation Forge entry point:
  `1670f7b023dcf110259ea39b63ba1a55cb011521`
- Stage 3 freeze commit:
  `a3bd09a0fcbc846c7b33b6c720eda96d136da87a`
- Annotated freeze tag: `stage3-generation-frozen-v1`
- HEG: `fd97451b0f3d87400d1d955a2c6b1b18303344ff`,
  read-only and clean
- Installed Codex CLI: `0.146.0`
- Frozen model: `gpt-5.6-luna`
- Frozen reasoning effort: `high`
- Freeze artifact SHA-256:
  `1a1fe217f7d4589304c0c777e9a86efe8ba0e198104b7ee5c6e56437a5c59e7a`
- Frozen config stable hash:
  `fe9836a63386d8bef7210671fffc2fb664b0b92abc16a5db82cf0d342ba602ad`
- Frozen config file SHA-256:
  `d697fae5973bc0fb9368941a983f33e2652f3fe241a0b48760b40d7f1d898333`
- Development manifest canonical SHA-256:
  `6d6cb608a291c2ac302dfd2657e84d70a1ddd955889f3b8fd77eaacb534cf5c9`
- Development manifest file SHA-256:
  `278eb0b02890c4b8d8a427c9345539b5558daff2206deb4958fa01bedbfcd664`
- Prompt bundle SHA-256:
  `89f94e35c489ee8fd06dab962c313e9a284fa3cd0827cffed4c47997e27a8869`

The freeze commit and annotated tag were pushed, and the preregistration
evidence was recorded on issue #9 before the Phase 2 command ran.
`live model results observed: false` was true at freeze and remains true.

## Implemented offline path

The frozen implementation contains:

- a thin adapter for the installed bidirectional JSON-RPC App Server protocol;
- fresh private Codex, SQLite, working, and temporary directories;
- strict startup configuration, schema negotiation, skill discovery and
  disablement, empty capability/runtime roots, read-only sandboxing, approval
  denial, server-request denial, bounded protocol framing, and process-group
  cleanup;
- installed `model/list` verification with provider fallback disabled;
- schema-derived system/request prompts and strict
  `stage3.generated_policy.v1` structured output;
- eight ordered, non-overlapping one-shot briefs;
- an eight-thread initial wave and a separate bounded repair wave;
- exact token-field capture without estimates;
- unchanged Stage 2A validation, normalized-AST identity, fixed behavior
  probes, deduplication, persistent worker execution, 10,000-call smoke, and
  model-free replay;
- an immutable 128-episode development manifest;
- independent policy trajectories, selected-plan-only scoring, deterministic
  eight-process evaluation, physical-core affinity, one-thread numerical
  libraries, reduction, bootstrap, champion selection, twelve named gates,
  and a complete second evaluation replay;
- bounded, redacted, atomic artifacts and canonical Rich/JSON rendering.

The App Server, generation, and policy runtimes contain no experimental
network path. Model code receives no repository, HEG, shell, filesystem,
browser, MCP/plugin, application, benchmark, baseline-result, oracle, or
hidden-result authority.

## Frozen campaign and resource budgets

- initial turns: exactly 8;
- initial concurrency: 8;
- repairs: at most 1 schema/AST-only turn per slot;
- maximum accepted live turns: 16;
- request/response: 65,536 / 16,384 bytes;
- event/stdout/stderr: 65,536 bytes each;
- transcript: 262,144 bytes;
- turn/campaign wall limits: 120 / 1,800 seconds;
- App Server address space: 2 GiB;
- App Server CPU/file/open-file/process limits:
  120 seconds / 8 MiB / 256 / 1,024;
- Stage 2A worker limits: unchanged;
- evaluation: at most 8 CPU workers, with 8 of 16 physical cores reserved;
- numerical-library threads: 1.

Every final App Server usage record must contain input, cached-input,
cache-write-input, output, reasoning-output, and total tokens. The installed
protocol does not expose a supported client token cap, so no cap or price was
invented.

## Development manifest and gates

The checked-in manifest freezes 128 paired episodes:

- orders 10 and 12;
- graph seeds 301–304;
- policy seeds 3001–3016;
- horizon 32;
- eight deterministic shards of 16 episodes.

The seeds are disjoint from the Stage 2C and Stage 2D development/diagnostic
seeds. The twelve frozen Stage 3 gates cover dependency provenance,
protocol/capsule safety, campaign authority, exact usage, minimum unique
yield, baseline AST distinctness, the two development thresholds, replay,
invalid graphs and worker failures, selected-only/equal/bounded accounting,
and repository/HEG validation.

Because authentication prevented the first live turn, candidate yield,
development metrics, champion selection, primary/replay evaluation hashes,
and scientific gates are all **not observed**, not failed.

## Authentication diagnosis

The normal user profile reports an existing ChatGPT login backed by the normal
Codex authentication file. A fresh private `CODEX_HOME` is unauthenticated
under every supported credential-store selector. The installed CLI exposes no
supported independent auth-store path/reference.

An exploratory symlink can make a private home read the normal auth file, but
that would cross the frozen isolation boundary and is neither documented nor
approved. Mutation Forge did not use it. It did not read, copy, parse, print,
store, bind-mount, or symlink authentication material, and it did not
introduce a direct API-key path.

The no-inference doctor still completed protocol initialization, disabled
bundled skills, queried `model/list`, verified exact
`gpt-5.6-luna`/`high`, and recorded 16 physical cores. Only private-capsule
authentication readiness failed.

## Phase 2 terminal artifact

Command:

```console
uv run mforge stage3 generate \
  --config configs/stage3-generation.toml \
  --concurrency 8 \
  --json
```

Durable run:

```text
runs/stage3-development/stage3-generation-fe9836a63386/
```

Terminal evidence:

- status: `infrastructure_failure`;
- failure code: `private_capsule_auth_unavailable`;
- provider calls: 0;
- model calls: 0;
- initial turns: 0;
- repair turns: 0;
- live model results observed: false;
- decision: `INCONCLUSIVE_INFRASTRUCTURE_FAILURE`.

The artifact retains the freeze, environment, configuration, prompt hashes,
development manifest, App Server status, generation summary, gate decision,
and terminal run summary. No evaluation was started after the generation
precondition failed.

## Validation

The immutable Phase 1 freeze passed:

```text
pytest:             222 passed, zero skipped
Ruff:               PASS
strict mypy:        PASS
mforge doctor:      PASS
git diff --check:   PASS
HEG pin/clean:      PASS
App Server offline: PASS
private auth ready: FAIL (expected fail-closed precondition)
```

Fake-server coverage includes initialization, model discovery, skill
disablement, terminal failure/cancellation/interruption, crash, malformed and
oversized protocol data, missing usage, unknown notification, and
server-initiated tool/request denial. Coordinator-health, bounded artifact,
redaction, concurrency, repair barrier, strict schema, deduplication,
Stage 2A execution, replay, seed, trajectory, reduction, gate-boundary, and
Rich/JSON parity tests pass.

## Scope preservation

Stage 2B remains the retained historical `NO_GO`. Stage 2C and Stage 2D
evidence are unchanged. HEG is unchanged. No candidate, benchmark outcome,
champion, evolution/archive search, full proposer, held-out evaluation, HEG
policy integration, or Stage 4 work was started.

## Decision

**`INCONCLUSIVE_INFRASTRUCTURE_FAILURE`**

Do not start Stage 4. A future retry requires explicit user authorization and
a supported way to authenticate an isolated Codex home. The frozen campaign
must not be silently changed or rerun with copied credentials, a symlinked
normal auth store, a substituted model/profile, modified prompts, extra
samples, or relaxed isolation.
