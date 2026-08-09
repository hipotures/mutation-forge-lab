# Current project status

This document is the canonical operational and scientific status for Mutation
Forge Lab as of the M10 repository consolidation.

## Canonical code

- The canonical M10 implementation snapshot on `main` is commit
  `631f7db5c37bd344f01575556a3cfbd563e743da`, tree
  `026c08b3df91232a5220667972e50a9e7b6420cd`. The document-only publication
  merge is a descendant of that snapshot and is recorded in consolidation
  issue
  [#64](https://github.com/hipotures/mutation-forge-lab/issues/64).
- M8 was merged by
  [PR #60](https://github.com/hipotures/mutation-forge-lab/pull/60):
  head `3d1a8794f6ac13c650d271d8d2a463ac848fd2f0`, normal merge commit
  `60285f623fa0f0670518337d114d0e52866926f2`.
- M9 was merged by
  [PR #62](https://github.com/hipotures/mutation-forge-lab/pull/62):
  head `25520eb515148827d42481a62daf7738ecd0907f`, normal merge commit
  `3b8363f1af15e7751085775cfce810fe68ba17f1`.
- M10 was merged by
  [PR #65](https://github.com/hipotures/mutation-forge-lab/pull/65):
  head `5023a3b78a158a9dd6beed8b77bf175bda1c8858`, normal merge commit
  `631f7db5c37bd344f01575556a3cfbd563e743da`.
- Native v2 remains the default experiment protocol.
- Ordinary-Python mode remains explicit opt-in through the
  `native-v3-python-v1` protocol and its dedicated scientific configuration.

## Implemented

- ordinary-Python policy generation;
- static AST validation and identity;
- isolated sandbox execution;
- safe relation-aware graph API;
- HEG scoring and the exact-verification seam;
- root/child multi-generation evolution;
- deterministic lineage and Search Memory;
- bounded provider concurrency;
- immediate evaluator queueing;
- Rich dashboard projection;
- JSON status projection with the same canonical counters;
- repository, configuration, program, graph, and result provenance;
- resume without repeating terminal work;
- removal of the superseded JSON DSL.

## Scientific results to date

No verified counterexample has been found. A heuristic fitness value is a
development result, not an exact certificate.

M9 completed at its frozen `provider_turn_budget` condition. It planned 64
slots, recorded 56 terminal slots and 8 pending generation-7 slots, evaluated
48 programs, and recorded 8 contract-invalid programs. It consumed 556,439
provider tokens. Its best development candidate was fresh root
`g0004-slot-04`, program
`485e744c0fbe5666a4224317aaf487424907fcd5af0553e6646b7fce5abcd142`,
with exact fitness `316968447/541220632`
(`0.585654774151329840655446409515`). M9 made zero exact-verifier
submissions and produced no verified result.

M10 is intentionally paused for provider budget with 17 terminal slots and 7
resumable pending slots. It consumed 121,090 provider tokens. Its best
development candidate is root `g0000-slot-01`, program
`19f705303dfab825ea5c7f43a118743f87f98d0b912bd9283dddfaba5f31f853`,
with exact fitness `157427937/270610316`. M10 has made zero exact-verifier
submissions, has zero exact-verifier results, and has not produced a verified
counterexample.

## Paused campaign

- State: `PAUSED_FOR_BUDGET`; `resumable = true`.
- Durable workspace:
  `/home/user/DEV/mutation-forge-lab-evidence/m10/sustained/native-v3-python-m10-sustained-v1`.
- Checkpoint:
  `/home/user/DEV/mutation-forge-lab-evidence/m10/sustained/native-v3-python-m10-sustained-v1/m10-repository-stabilization-checkpoint.json`
  (SHA-256
  `ed8519e34ee5437584f6c15cc0df91ca468405ca77ae6e33b6ce55052dfccd32`).
- Pause record:
  `/home/user/DEV/mutation-forge-lab-evidence/m10/emergency-stop/paused-for-budget.json`
  (SHA-256
  `8886958cf4aa15351f80777f71a4833aa83009cd8794f372f884db9b49dbaf0e`).
- Frozen campaign code: branch `native-v3-python-provider-throughput`, commit
  `de2ff1be3669cade2fc347317c3ed80070afa279`, tree
  `aa1c40ece6b252c0f05e63fa2c50219b3e3e79bd`, preserved by annotated tag
  `native-v3-python-m10-paused`.
- Frozen configuration SHA-256:
  `85eb731bec9d226a1862605ee17f4fe21d4d56ee342da61c229bf4d80ddfd07e`.
- Canonical workspace-manifest SHA-256:
  `c36a2616ae73451eb6ea897082febcdba0e9a4162c9cf87f805960d3977d35b8`.
- Terminal slots: 17 (14 evaluated and 3 provider failures).
- Resumable pending slots: 7 (4 interrupted in flight and 3 unstarted).
  Interrupted and pending slots are not scientific failures.
- Provider token usage: 98,780 input, 11,264 cached input, 22,310 output,
  10,085 reasoning output, 121,090 total.
- Exact resume command:

  ```console
  uv run mforge experiment run --config configs/scientific/native-v3-python-m10-v1.toml --json
  ```

The campaign must not be resumed without new, explicit provider-budget
authorization. Resume must use the frozen scientific configuration and must
preserve existing terminal work.

## Dashboard operation

Read-only Rich inspection:

```console
uv run mforge experiment status --config /home/user/DEV/mutation-forge-lab-evidence/m10/sustained/native-v3-python-m10-sustained-v1/python-preview.toml --pause-record /home/user/DEV/mutation-forge-lab-evidence/m10/emergency-stop/paused-for-budget.json --dashboard
```

Read-only JSON inspection:

```console
uv run mforge experiment status --config /home/user/DEV/mutation-forge-lab-evidence/m10/sustained/native-v3-python-m10-sustained-v1/python-preview.toml --pause-record /home/user/DEV/mutation-forge-lab-evidence/m10/emergency-stop/paused-for-budget.json --json
```

Future authorized resume:

```console
uv run mforge experiment run --config configs/scientific/native-v3-python-m10-v1.toml --json
```

The status commands read the existing workspace, display
`PAUSED_FOR_BUDGET`, and do not schedule provider or evaluator work. The
consolidation smoke reported the same Rich and JSON counters: generation 2,
17 of 24 terminal, 7 resumable pending, 22 provider reservations, 22 provider
candidates, 23 provider turns, concurrency 4 with 0 active and peak 4, 14
evaluator completions, 12 configured evaluators with 0 active and peak 2,
121,090 tokens, the best program and exact fitness above, and exact-verifier
state 0 submissions, 0 results, not verified.

## Remaining work

1. Resume the paused M10 campaign when budget is available.
2. Analyze results after completion.
3. Decide whether to run a new frozen scientific campaign.
4. Separately decide whether Native v2 should ever cease to be default.
5. Revisit later scientific roadmap issues only after reviewing completed
   campaign evidence.

## Repository cleanup

- Before consolidation: 26 local branches, 21 remote branches including
  `main`, and 17 worktrees.
- After consolidation: one local branch (`main`), one remote branch (`main`),
  and one canonical worktree (`/home/user/DEV/mutation-forge-lab`). The short
  document publication branch exists only for review and is deleted after its
  merge.
- Required retained annotated tags:
  `native-v3-python-m7-complete`, `native-v3-python-m8-merged`,
  `native-v3-python-m9-merged`, `native-v3-python-m10-paused`, and
  `native-v3-python-m10-code-merged`. Earlier frozen scientific tags are also
  retained.
- Consolidation evidence:
  `/home/user/DEV/mutation-forge-lab-evidence/repository-consolidation`.
- Frozen M10 evidence:
  `/home/user/DEV/mutation-forge-lab-evidence/m10`.
- Unique superseded branch history is preserved in
  `/home/user/DEV/mutation-forge-lab-evidence/repository-consolidation/superseded-branches.bundle`.
- The two user-local `experiment.toml` changes are preserved as patches in
  `/home/user/DEV/mutation-forge-lab-evidence/repository-consolidation/dirty-worktrees`;
  neither was committed.
- After the document publication PR and consolidation issue close, there are
  no open PRs and the sole remaining open tracking issue is
  [#66](https://github.com/hipotures/mutation-forge-lab/issues/66), which
  carries no authorization to resume the campaign automatically.
