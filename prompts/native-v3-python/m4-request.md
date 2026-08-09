# Objective

Write one reusable ordinary-Python policy for a scientific search related to
the Erdős–Gyárfás conjecture. The search studies connected graphs of minimum
degree at least three and tries to eliminate cycles whose lengths are powers of
two. The host measures bounded component evidence for the active forbidden
lengths and accepts a rewrite only when its conservative energy interval is
proved strictly better than the incumbent interval.

Your policy proposes a graph rewrite; it never computes or verifies the score.
Prefer deterministic structural logic with a meaningful fallback. A proposal
may be rejected even when it is structurally legal, so returning `NoPlan` is
valid when the policy cannot justify a candidate.

# Required response

Return exactly one JSON object with exactly two fields:

```json
{
  "schema_version": "mforge.native.python_policy_response.v1",
  "source": "<complete ordinary Python source>"
}
```

The source must define exactly this public entry point:

```python
def propose(ctx, graph, api, seed):
    ...
```

The return annotation may be omitted. If present, it must be exactly:

```python
RewritePlan | NoPlan
```

# Visible immutable values

`ctx` exposes only:

- `step_index`
- `horizon`
- `acceptance_profile_id`
- `stagnation_steps`
- `exploration_window_index`
- `accepted_rewrites`
- `accepted_non_improving_rewrites`
- `consecutive_non_improving_rewrites`
- `witness_cap`
- `invocation_ordinal`
- `forbidden_lengths`, an increasing tuple of active cycle lengths

`graph` exposes only:

- `order`
- `edge_count`
- `minimum_degree`
- `maximum_degree`

Graph references returned by the API are opaque and valid only during the
current invocation. They reveal no vertex names, adjacency, or complete edge
list.

# Safe graph API

Every selector returns a deterministic tuple of at most 64 opaque references.
`mode` must be `"min"` or `"max"` and returns all tied extremes.

```python
api.vertices_degree_extreme(mode="max")
api.vertices_degree_class(degree)
api.vertices_witness_load_extreme(length, mode="max")
api.edges_witness_load_extreme(length, mode="max")
api.vertices_articulation_risk(mode="max")
api.edges_bridge_risk(mode="max")
api.edges_removable()
api.vertices_distance_band(source, minimum, maximum)
api.non_edges_from_vertex(vertex)
api.non_edges_legal()
api.non_edges_local_cycle_risk(mode="max")
api.paths_length_two()
api.matching_k_switch_reconnections(k)
api.relocations_legal()
api.edge_fanouts_legal()
```

Semantics:

- degree selectors use current private-overlay degrees;
- witness-load selectors accept only a value from
  `ctx.forbidden_lengths`; they rank host-supplied sampled loads, with absent
  loads treated as zero;
- articulation and bridge selectors rank current binary structural risk;
- `edges_removable()` returns current edge references; removal is still
  subject to final validation;
- distance bands use inclusive nonnegative distances;
- `non_edges_legal()` returns absent-edge references;
- local-cycle risk ranks absent edges by their number of common neighbours;
- length-two paths identify two incident edges and their possible fold;
- matching reconnections accept only `k` equal to 2, 3, or 4;
- relocation and fanout selectors return syntactically applicable candidates;
  final graph validity is checked only by `emit()`.

Choose reproducibly with:

```python
api.pick(items, seed, salt, feature="uniform")
```

The supplied `seed` must be passed unchanged. `salt` is a printable string or
integer chosen by your policy. `feature` is `"uniform"`, `"degree"`, or
`"inverse_degree"`; degree weighting is valid only for vertex references.
Empty input returns `None`.
Test that case with `if not selected:`; identity comparisons such as
`selected is None` are outside the accepted AST subset.

Actions update a private overlay:

```python
api.add_edge(non_edge)
api.remove_edge(edge)
api.relocate_endpoint(relocation)
api.k_switch(matching)
api.edge_fanout(fanout)
api.edge_fold(path)
```

Finish with exactly one host-minted terminal result:

```python
api.emit()
api.no_plan(reason="EXPLICIT")
```

`emit()` computes the net rewrite, rejects no-effect results, requires a
connected final graph with minimum degree at least three, and asks the trusted
host to validate the candidate. It returns either a host-minted `RewritePlan`
or `NoPlan("NO_EFFECT" | "ILLEGAL_FINAL_STATE")`.

Allowed explicit no-plan reasons are `"EXPLICIT"`, `"NO_MATCH"`,
`"ILLEGAL_FINAL_STATE"`, and `"NO_EFFECT"`.

# Python restrictions

- Use only ordinary local variables, conditionals, helper functions, and
  statically bounded loops.
- Helper names must start with `helper_`; use at most 16 helpers.
- Do not use imports, classes, async code, generators, exceptions, context
  managers, lambdas, comprehensions, recursion, reflection, dynamic attribute
  access, global or nonlocal state, default arguments, variadic arguments,
  decorators, or type annotations other than the optional exact return
  annotation above.
- Do not call `eval`, `exec`, `compile`, `open`, `print`, or any function not
  documented here.
- Use only bounded literal tuples, lists, and dictionaries.
- Every `for` loop must be statically bounded by a documented selector result,
  a bounded literal, or `range(...)` with at most 64 trips.
- Do not construct `RewritePlan` or `NoPlan` directly.
- Do not attempt to access files, environment variables, clocks, processes,
  network resources, scoring, verification, or host state.

Return a complete policy with substantive selection, fallback, and terminal
logic. Do not return pseudocode, Markdown, or a declarative graph-program AST.
