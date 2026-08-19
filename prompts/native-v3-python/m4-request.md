# Objective

Write one reusable ordinary-Python mutation policy for a scientific search related to the Erdős–Gyárfás conjecture: connected graphs of minimum degree at least three, trying to eliminate cycles whose lengths are powers of two. The host scores and accepts rewrites; the policy only proposes them and never computes or verifies score. Prefer deterministic structural logic with meaningful fallback; `NoPlan` is valid when no candidate is justified.

# Required response

Return exactly:
```json
{"schema_version":"mforge.native.python_policy_response.v1","source":"<complete ordinary Python source>"}
```

The source must define `def propose(ctx, graph, api, seed): ...`. Its return annotation may be omitted; if present it must be exactly `RewritePlan | NoPlan`.

# Invocation snapshots

`ctx` and `graph` are immutable invocation-start snapshots available under exactly these attributes.

`ctx`: `step_index`, `horizon`, `acceptance_profile_id`, `stagnation_steps`, `exploration_window_index`, `accepted_rewrites`, `accepted_non_improving_rewrites`, `consecutive_non_improving_rewrites`, `witness_cap`, `invocation_ordinal`, `forbidden_lengths` (increasing tuple of active cycle lengths).

`graph`: `order`, `edge_count`, `minimum_degree`, `maximum_degree`.

Actions change a private graph overlay, not these snapshot values. Subsequent API selectors observe the current overlay including earlier actions from the same invocation.

API graph references are opaque and invocation-local. They reveal no vertex names, edge endpoints, adjacency, selector score, or complete edge list. Do not inspect, decode, structurally compare, or infer graph information from their representation.

# Selector rules

Each selector returns a deterministic tuple of at most 64 opaque references and may return a bounded host-generated subset rather than an exhaustive set. `mode` is `"min"` or `"max"` and returns all tied references at that extreme among considered values. Tuple order exists only for reproducibility and has no graph-theoretic meaning; never use position, prefix/suffix, adjacency in the tuple, runs, spans, alternation, parity, or other index patterns as graph features.

Extreme selectors expose references, not the numeric extreme. `mode="max"` does not imply a positive or large value: all-zero witness loads may make all considered references tie; binary articulation/bridge risk may make all vertices/edges tie when none has risk; equal common-neighbour counts may make all considered non-edges tie.

# Safe graph API

```python
# all tied vertices having minimum or maximum current-overlay degree
api.vertices_degree_extreme(mode="max")

# vertices having exactly `degree` in the current overlay
api.vertices_degree_class(degree)

# vertices tied at min/max sampled witness load for active forbidden `length`; missing loads=0, numeric load hidden
api.vertices_witness_load_extreme(length, mode="max")

# edges tied at min/max sampled witness load for active forbidden `length`; missing loads=0, numeric load hidden
api.edges_witness_load_extreme(length, mode="max")

# vertices tied at min/max binary articulation indicator: 1 articulation, 0 otherwise
api.vertices_articulation_risk(mode="max")

# edges tied at min/max binary bridge indicator: 1 bridge, 0 otherwise
api.edges_bridge_risk(mode="max")

# all current edges; deletion is not guaranteed to preserve connectivity, minimum degree, or final validity
api.edges_removable()

# vertices at inclusive current-overlay shortest-path distance [minimum,maximum] from `source`, with 0<=minimum<=maximum
api.vertices_distance_band(source, minimum, maximum)

# absent edges incident to `vertex` in the current overlay
api.non_edges_from_vertex(vertex)

# all current absent edges; adding one is not guaranteed to improve or preserve forbidden-cycle structure
api.non_edges_legal()

# absent edges tied at min/max number of common neighbours of their endpoints; numeric count hidden
api.non_edges_local_cycle_risk(mode="max")

# opaque current paths u-center-v of length two, identifying a possible fold without revealing vertices
api.paths_length_two()

# bounded host-generated k-switch candidates, k in {2,3,4}: remove k pairwise vertex-disjoint edges, add k different edges reconnecting exactly the same 2k endpoints
api.matching_k_switch_reconnections(k)

# same, with supplied current `edge` included among removed source edges
api.matching_k_switch_reconnections_for_edge(edge, k)

# endpoint relocations: current (u,v) -> (u,w) or (v,w), removing (u,v), preserving one endpoint, replacement edge currently absent; choice opaque
api.relocations_legal()

# same relocation selector restricted to supplied current `edge`
api.relocations_legal_for_edge(edge)

# fanouts: current (u,v) -> (u,w)+(v,w), removing (u,v); both additions currently absent; choice opaque
api.edge_fanouts_legal()

# same fanout selector restricted to supplied current `edge`
api.edge_fanouts_legal_for_edge(edge)
```

Relocation and fanout candidates are only syntactically applicable; final connectivity and minimum degree are checked by `emit()`.

# Reproducible choice

```python
api.pick(items, seed, salt, feature="uniform")
```

Pass `seed` unchanged. `salt` is a printable string or integer chosen by the policy. `feature` is `"uniform"`, `"degree"`, or `"inverse_degree"`; degree-based features are valid only for vertex references and use current-overlay degree. Empty input returns `None`; test with `if not selected:` because identity comparisons such as `selected is None` are outside the accepted AST subset.

# Actions

```python
# add supplied currently absent edge
api.add_edge(non_edge)

# remove supplied current edge; action alone does not guarantee connectivity or minimum degree
api.remove_edge(edge)

# selected relocation: remove (u,v), add either (u,w) or (v,w), preserving exactly one endpoint
api.relocate_endpoint(relocation)

# remove candidate's k source edges and add its k reconnection edges on exactly the same endpoint set
api.k_switch(matching)

# selected fanout: remove (u,v), add (u,w)+(v,w); degrees u,v unchanged, degree w +2, edge count +1
api.edge_fanout(fanout)

# selected fold u-center-v: remove (u,center)+(center,v), add absent (u,v); degrees u,v unchanged, center -2, edge count -1
api.edge_fold(path)
```

Actions update the private overlay. References can become stale after overlay edits; obtain selector results from the overlay state on which they will be used.

# Terminal result

Finish with exactly one host-minted terminal result:
```python
api.emit()
api.no_plan(reason="EXPLICIT")
```

`emit()` computes the net rewrite relative to the input graph, rejects no-effect results, requires a connected final graph with minimum degree at least three, and asks the trusted host to validate the candidate. It returns a host-minted `RewritePlan` or `NoPlan("NO_EFFECT" | "ILLEGAL_FINAL_STATE")`.

Allowed explicit no-plan reasons: `"EXPLICIT"`, `"NO_MATCH"`, `"ILLEGAL_FINAL_STATE"`, `"NO_EFFECT"`.

The host alone evaluates scientific score and acceptance. Do not infer unexposed score from reference order, opaque tokens, or undocumented behavior.

# Python restrictions

- Use only ordinary local variables, conditionals, helper functions, and statically bounded loops.
- Helper names start with `helper_`; at most 16 helpers.
- Functions may be defined only at module top level: exactly one `propose` plus optional `helper_*` functions. Nested function definitions are forbidden.
- Helper parameters must not be named `ctx`, `graph`, `api`, `seed`, any defined function name, any allowed built-in, `RewritePlan`, or `NoPlan`. Helpers cannot call the Safe Graph API; every `api.*` call must occur directly inside `propose`.
- Allowed built-ins only: `abs`, `all`, `any`, `bool`, `enumerate`, `int`, `len`, `max`, `min`, `range`, `reversed`, `sum`, `tuple`.
- No imports, classes, async, generators, exceptions, context managers, lambdas, comprehensions, recursion, reflection, dynamic attribute access, global/nonlocal state, default arguments, variadic arguments, decorators, or type annotations except the optional exact return annotation above.
- Do not call `eval`, `exec`, `compile`, `open`, `print`, or any undocumented function.
- Use only bounded literal tuples, lists, and dictionaries.
- Every `for` loop must be statically bounded by a documented selector result, a bounded literal, or `range(...)` with at most 64 trips.
- Do not construct `RewritePlan` or `NoPlan` directly.
- Do not access files, environment variables, clocks, processes, network resources, scoring, verification, or host state.

Return complete code with substantive structural selection, a scientifically motivated mutation strategy, meaningful fallback logic, and exactly one terminal result on every path. No pseudocode, Markdown, or declarative graph-program AST.
