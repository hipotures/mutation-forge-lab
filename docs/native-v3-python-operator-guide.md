# Native v3 ordinary-Python preview operator guide

## What this preview is

Mutation Forge has two experiment routes:

- Native v2 is the production default. A normal `experiment.toml` without a
  `protocol` field continues to run Native v2.
- The ordinary-Python preview is an explicit, bounded two-generation route.
  It is selected only by `protocol = "native-v3-python-v1"` in a separate
  preview configuration.

The preview asks Codex App Server for ordinary Python source implementing:

```python
def propose(ctx, graph, api, seed):
    ...
```

The host validates the source without executing it, computes its source and
canonical-AST identities, and sends it to an isolated worker. The worker is the
only process that compiles or executes generated source. It exposes a fixed
safe API and returns host-validated `RewritePlan`, `NoPlan`, or typed program
failure records. The serial scientific evaluator then uses the authoritative
HEG score worker, exact rational interval comparisons, and the exact-verifier
seam.

The population contract is fixed:

```text
generation 0: 8 fresh roots
generation 1: 4 exact-parent children + 4 fresh roots
```

Invalid, duplicate, missing, provider-failed, and evaluation-infrastructure
terminal slots consume their planned positions. Resume submits only pending
slots.

## Security boundary

M1 is a fail-closed AST validator. It rejects imports, reflection, dynamic
code, unapproved calls, and syntax outside the accepted policy subset. M1 does
not compile or execute candidate source.

M2 runs each accepted policy in the frozen Linux
`bubblewrap + namespaces + seccomp + rlimits` worker. The coordinator,
provider, evaluator, and test processes do not execute candidate source.
Pre-invocation worker rotation is transparent and preserves the frozen limits.

The preview does not contain or call the retired JSON-AST contract,
interpreter, IR compiler, schemas, prompts, or runtime dispatch. Old JSON-DSL
workspaces are unsupported and are rejected before provider or backend
construction. They cannot be upgraded or reinterpreted as Python workspaces.

## Create a fresh preview workspace

Copy the versioned example; never edit the repository's `experiment.toml`:

```bash
cp configs/examples/native-v3-python-preview-v1.toml \
  /home/user/DEV/mutation-forge-lab-evidence/my-preview.toml
```

Set a unique `exp_id`, a durable absolute `workspace`, and the correct sibling
HEG path. Acceptance runs require both repositories to be clean before App
Server starts.

Start the preview explicitly:

```bash
uv run mforge experiment run \
  --config /home/user/DEV/mutation-forge-lab-evidence/my-preview.toml \
  --json
```

The convenience launcher creates a fresh versioned configuration:

```bash
uv run python scripts/native_v3_python_m6_live_preview.py \
  --output-root /home/user/DEV/mutation-forge-lab-evidence/my-preview
```

Do not use `/tmp` for acceptance evidence.

## Inspect status

Status is read-only and never contacts the provider:

```bash
uv run mforge experiment status \
  --config /home/user/DEV/mutation-forge-lab-evidence/my-preview.toml \
  --json
```

The projection reports the active protocol and mode, generation and slot
counts, provider turns and usage, program identities and lineage, sandbox
telemetry, policy/API behavior, graph-score attempts, interval fitness,
recovery state, exact-verifier activity, and terminal/scientific status. It
does not expose source, provider thread identifiers, paths, credentials, or
raw transport payloads.

## Stop at a durable boundary and resume

While the foreground preview is running, request a resumable stop from another
terminal:

```bash
uv run python scripts/native_v3_python_m6_live_preview.py \
  --request-stop \
  --config /home/user/DEV/mutation-forge-lab-evidence/my-preview.toml
```

The running process stops before the next provider submission, after its
current durable candidate boundary. Wait for it to exit and confirm
`state=blocked`, `terminal_reason=operator_stop`, and `resumable=true`.

Resume the exact workspace with the same configuration:

```bash
uv run python scripts/native_v3_python_m6_live_preview.py \
  --config /home/user/DEV/mutation-forge-lab-evidence/my-preview.toml
```

Resume verifies repository, HEG, configuration, prompt, schema, runtime,
manifest, Search Memory, source, lineage, duplicate, evaluation, and panel
identities before continuing. A mismatch fails before App Server startup.

## Artifact locations

For `workspace = "/evidence/preview"` and `exp_id = "example"`, the experiment
root is `/evidence/preview/example`.

- Exact generated source:
  `generations/generation-NNNN/slot-NN/source.py`
- Candidate identity and status:
  `generations/generation-NNNN/slot-NN/candidate.json.gz`
- Scientific cases and semantic traces:
  `generations/generation-NNNN/slot-NN/evaluations/*.json.gz`
- Immutable generation allocation:
  `generations/generation-NNNN/manifest.json.gz`
- Host and source-free model Search Memory:
  `generations/generation-NNNN/search-memory.json.gz`
- Provider lifecycle artifacts:
  per-slot `provider-*` directories and `provider-runtime`
- Counterexample/verifier artifacts:
  `scientific-artifacts/<candidate>/<case>/counterexamples`
- Public state:
  `python-preview-state.json.gz`
- Terminal search report:
  `m5-report.json.gz`
- Resumable stop or infrastructure boundary:
  `m5-stop.json.gz`
- Acceptance provenance:
  `acceptance-provenance.json.gz`

Only the exact verifier can set `VERIFIED`. A heuristic zero is a submission,
not a verified counterexample. Status reports submissions, records, outcomes,
and the `exact_verifier_only` authority separately.

## Read the failure taxonomy

- `provider_failed`: App Server/provider transport or response production
  failed. No scientific result was created.
- `contract_invalid`: both the initial response and bounded repair failed M1.
  No worker or scientific result was created.
- `PROGRAM_FAILURE`: accepted source ran in M2 but returned an invalid value,
  timed out, crashed, or raised a candidate-owned exception.
- `evaluation_infrastructure_failure`: trusted sandbox host, backend, scorer,
  evaluator, or verifier infrastructure failed. It is not scientific success.
- `NO_PLAN`: the policy deliberately proposed no move; the step budget is
  consumed.
- Scientific rejection: a legal rewrite was scored but did not have a proved
  strictly better interval.
- `VERIFIED_COUNTEREXAMPLE`: an apparent zero passed the authoritative exact
  verification path.

A completed provider cohort is never reported as scientific success.

## Cleanup

Inspect status and retain any required evidence bundle first. Then remove only
the explicit experiment root:

```bash
rm -r -- /home/user/DEV/mutation-forge-lab-evidence/my-preview/example
```

Remove a generated convenience configuration separately:

```bash
rm -- /home/user/DEV/mutation-forge-lab-evidence/my-preview/configs/<file>.toml
```

Never recursively target a workspace parent, repository root, `$HOME`, or the
sibling HEG checkout. The preview has no JSON-DSL compatibility fallback; keep
old workspaces only as archived evidence.
