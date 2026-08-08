# Native v3 ordinary-Python M4 first model root

Status: M4 implementation and one-root live evidence. M5 is not started.

Accepted base: `2ab0c5789748b85e875224c4e210ff3dea0c94c8`.

## Scope

M4 adds an inactive, one-root path:

```text
Codex App Server
-> two-field ordinary-Python response
-> M1 validation and identity
-> M2 isolated worker
-> M3 serial evaluator
-> authoritative current HEG component evidence
-> exact-verification seam
```

It does not activate an experiment route. Native v2 remains the default.
There is no population, lineage, parent selection, thread fork, Search Memory,
generation loop, preview activation, concurrency pool, or DSL disconnection.
`experiment.toml` is unchanged.

## Provider contract

The model received a 152-byte system prompt, a 5,650-byte request prompt, and
a 401-byte two-field Structured Output schema. The response schema requires
only:

```json
{
  "schema_version": "mforge.native.python_policy_response.v1",
  "source": "<ordinary Python source>"
}
```

The request explains the Erdős–Gyárfás objective, immutable context and graph
views, all 24 accepted safe API methods, opaque reference semantics, terminal
results, and the M1 Python subset. Tests prove that model-facing text contains
no host hashes, UUIDs, workspace paths, provider bookkeeping, raw labels,
adjacency, scorer state, verifier state, or held-out data.

The installed App Server rejected the first preflight request before model
execution because `properties.schema_version` had `const` but no explicit
`type`. That request produced no response, usage, or model turn. M4 corrected
the schema to `type: "string"` and retained the failed preflight artifacts.

The subsequent and only actual model root used `gpt-5.6-luna` at `high`
reasoning effort. The turn completed in 163,953 ms without retry or repair.
Final usage was:

| Field | Value |
|---|---:|
| input tokens | 3,414 |
| cached input tokens | 0 |
| cache-write input tokens | 0 |
| output tokens | 8,974 |
| reasoning output tokens | 8,015 |
| total tokens | 12,388 |
| final | true |
| partial | false |

The legal Code Mode-disabled warning occurred once. The turn still ended with
`turn/completed.status=completed`, `error=null`, a final response, and exact
final usage.

## Exact generated source

```python
def propose(ctx, graph, api, seed):
    if not ctx.forbidden_lengths:
        return api.no_plan(reason="NO_MATCH")

    focus_length = ctx.forbidden_lengths[0]
    if ctx.exploration_window_index % 2 == 1:
        focus_length = ctx.forbidden_lengths[-1]

    hot_vertices = api.vertices_witness_load_extreme(focus_length, mode="max")
    hot_vertex = api.pick(hot_vertices, seed, "focus-vertex", feature="degree")
    local_band = ()
    if hot_vertex:
        local_band = api.vertices_distance_band(hot_vertex, 1, 2)

    hot_edges = api.edges_witness_load_extreme(focus_length, mode="max")
    hot_edge = api.pick(hot_edges, seed, "focus-edge", feature="uniform")

    switch_size = 2
    if ctx.stagnation_steps > 0:
        switch_size = 3
    if ctx.consecutive_non_improving_rewrites > 1:
        switch_size = 4
    if graph.maximum_degree - graph.minimum_degree > 3:
        switch_size = 4
    if local_band and ctx.accepted_non_improving_rewrites > ctx.accepted_rewrites:
        switch_size = 4

    action_taken = 0
    switch_candidates = api.matching_k_switch_reconnections(switch_size)
    selected_switch = api.pick(switch_candidates, seed, "witness-switch", feature="uniform")
    if not selected_switch and switch_size != 2:
        switch_candidates = api.matching_k_switch_reconnections(2)
        selected_switch = api.pick(switch_candidates, seed, "conservative-switch", feature="uniform")
    if hot_edge and selected_switch:
        api.k_switch(selected_switch)
        action_taken = 1

    if not action_taken and ctx.stagnation_steps > 0:
        switch_candidates = api.matching_k_switch_reconnections(2)
        selected_switch = api.pick(switch_candidates, seed, "fallback-switch", feature="uniform")
        if selected_switch:
            api.k_switch(selected_switch)
            action_taken = 1

    if not action_taken:
        fanouts = api.edge_fanouts_legal()
        selected_fanout = api.pick(fanouts, seed, "fanout-fallback", feature="uniform")
        if selected_fanout:
            api.edge_fanout(selected_fanout)
            action_taken = 1

    if not action_taken and graph.maximum_degree > graph.minimum_degree:
        relocations = api.relocations_legal()
        selected_relocation = api.pick(relocations, seed, "relocation-fallback", feature="uniform")
        if selected_relocation:
            api.relocate_endpoint(selected_relocation)
            action_taken = 1

    if not action_taken:
        removable = api.edges_removable()
        remove_edge = hot_edge
        if not remove_edge:
            remove_edge = api.pick(removable, seed, "removable-edge", feature="uniform")

        addition_pool = ()
        if hot_vertex:
            addition_pool = api.non_edges_from_vertex(hot_vertex)
        if not addition_pool:
            addition_pool = api.non_edges_legal()
        if ctx.stagnation_steps > ctx.exploration_window_index:
            risky_additions = api.non_edges_local_cycle_risk(mode="max")
            if risky_additions:
                addition_pool = risky_additions

        add_edge = api.pick(addition_pool, seed, "edge-reroute", feature="uniform")
        if remove_edge and add_edge:
            api.add_edge(add_edge)
            api.remove_edge(remove_edge)
            action_taken = 1

    if not action_taken:
        paths = api.paths_length_two()
        selected_path = api.pick(paths, seed, "fold-fallback", feature="uniform")
        if selected_path:
            api.edge_fold(selected_path)
            action_taken = 1

    if not action_taken:
        return api.no_plan(reason="NO_MATCH")
    return api.emit()
```

This is substantive ordinary Python, not pseudocode or a lowered JSON DSL.

## M1 identity

Validation passed with no diagnostics:

| Field | Value |
|---|---|
| source bytes | 3,622 |
| AST nodes | 631 |
| helper functions | 0 |
| source SHA-256 | `54dddd80c443aea2049f49c3a5c0189641452a2817850f190729ea903581585a` |
| canonical AST SHA-256 | `22b9c0e94aba485f1dcce1a4fbf96012be37d90ae3761a11ef4b6ec4d54b3dcd` |
| program hash | `ff19d2a8c6a19caa111cc8fc57bf23fa14dfffe779753d1d8fd8048a32d3daf7` |

The archived source is byte-identical to the exact provider response.

## M2 sandbox

The policy ran only in the accepted M2 worker. The worker performed one call,
had no failure or rotation, started in 27.37 ms, reached 19,180 KiB RSS, and
captured no stderr. It attested an empty `/work`, exact sanitized environment,
only file descriptors 0/1/2, user/mount/network/PID/IPC/UTS namespaces,
`no_new_privileges`, seccomp probes, and the frozen rlimits.

Frozen limits remained unchanged: 1-second `propose`, 256 MiB address space,
60-second invocation-eligibility process age, graph order 128, 256 total API
calls, 64 selector calls, 64 action calls, 64 selector results, 2,048 random
draws, 4,096 loop entries, 256 helper invocations, helper depth 8, and the
accepted frame/resource limits.

## M3 scientific result

The invocation produced a host-minted `REWRITE_PLAN`:

```text
removed: (16,17), (22,23)
added:   (16,23), (17,22)
operator_family: native_v3_python_policy
```

The current HEG C++ scorer made two authoritative score attempts:

| Evidence | Before | After |
|---|---:|---:|
| witness interval | [85, 85] | [82, 82] |
| weighted penalty interval | [440, 440] | [424, 424] |
| energy interval | [59,762,395, 59,762,395] | [57,652,950, 57,652,950] |

The strict improvement was proved and the rewrite was accepted. The exact
fitness interval was:

```text
[153194971 / 270610316, 153194971 / 270610316]
```

The semantic trace hash is
`693b7f5636113ea8984730abee058913c4b679ae5d665c0cb057d6ee484fca81`.
The behavior signature is
`eb92ac62372875939dd857919f4470df718c9afb2583cb8f01ab6921084e16dc`.

Neither evidence vector was zero, so the exact verifier received zero
submissions. The report records `exact_verifier_only` as the sole authority;
no heuristic result was promoted to `VERIFIED`.

## Artifact recovery and replay

The live turn initially reached a post-evaluation artifact error because the
local transport had final usage in memory/provider-raw but the M4 orchestrator
had not materialized the duplicated `slot-00.usage.json.gz` expected by the
frozen turn contract. M4 now materializes that exact final usage before
indexing. A regression test exercises this case through the fake App Server.

The exact retained root was then replayed with zero model, provider, App
Server, or network calls. Recovery reran M1, re-attested the M2 worker, and
reproduced M3/HEG behavior. It preserved:

- the original incomplete manifest as `turn-manifest.attempt-01.json.gz`;
- the original failure report as `m4-report.pre-recovery.json.gz`;
- the exact live provider events, response, final usage, warning, duration,
  and source.

The final manifest is complete with no missing files, exact final usage, and
completed validation. The exact source archive and scientific evaluation live
outside the immutable provider-turn tree. Recovery refuses a second run.

Live evidence root:

```text
/tmp/mutation-forge-native-v3-python-m4/native-v3-python-m4-20260808T092237Z
```

## Verification gates

Before the live root:

- Ruff passed.
- mypy passed.
- 237 focused M1-M4, App Server, and parity tests passed.
- the frozen App Server artifact fixture had 131 files and 7 parity tests
  passed.
- a recorded M4 provider response passed through the current HEG C++ scorer.
- the sibling HEG repository remained clean at
  `27cbec9c2307b6ea5f936f858821d11d808b68f3`.

After final implementation:

- Ruff passed.
- mypy passed for 163 source files.
- the final focused M1-M4/App Server/HEG suite passed.
- the Native v2 real-provider smoke passed with one initial turn, zero repair
  turns, exact final usage of 10,102 tokens, a complete 25-file turn manifest,
  and a match to the frozen App Server structural profile.
- the full suite collected 1,027 tests: 1,000 passed, 25 failed, and 2 errored.
  M4 added 19 passing tests. The failure and error node-ID sets are exactly
  equal to the accepted M3 baseline, with no new or resolved IDs. They remain
  attributable to the known HEG pin/dashboard/stage-fixture baseline.
- `experiment.toml` remained unchanged.

## Known limitations

- M4 is Linux-only because the accepted M2 isolation uses bubblewrap,
  namespaces, seccomp, and rlimits.
- The Code Mode-disabled notification remains a recorded legal warning; it
  does not relax any success condition.
- M4 evaluates one root on one accepted episode. It makes no claim about
  population quality, generalization, selection, or evolution.
- The recovery utility is deliberately single-use and accepts only the
  retained exact M4 provider response and final usage. It cannot contact a
  model or repair a policy.
- Native v3 ordinary-Python preview remains inactive. M5 and later issues are
  not authorized by this work.
