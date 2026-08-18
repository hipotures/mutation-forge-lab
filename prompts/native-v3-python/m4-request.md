# Objective

Write one reusable ordinary-Python policy for a scientific search related to
the Erdős–Gyárfás conjecture. The search studies connected graphs of minimum
degree at least three and tries to eliminate cycles whose lengths are powers of
two. The host measures bounded component evidence for the active forbidden
lengths and accepts a rewrite only when its conservative energy interval is
proved strictly better than the incumbent interval.

Your policy proposes a graph rewrite; it never computes or verifies the score.
Prefer deterministic structural logic with a meaningful fallback. A proposal
may be rejected even when it is structurally applicable, so returning `NoPlan`
is valid when the policy cannot justify a candidate.

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

# Invocation snapshots

`ctx` and `graph` are immutable snapshots for one invocation of `propose`.
They are available under exactly the attribute names listed below.

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

The four `graph` values describe the input graph at the start of the invocation.
They do not change after actions. Actions modify a private graph overlay instead.
All subsequent API selectors observe the current private overlay, including changes
made by earlier actions in the same invocation.

Graph references returned by the API are opaque and valid only during the current
invocation. They reveal no vertex names, edge endpoints, adjacency, selector score,
or complete edge list. Do not inspect, compare structurally, decode, or infer graph
information from the representation of an opaque reference.

# Selector result rules

Every selector returns a deterministic tuple of at most 64 opaque references.
The tuple may be a bounded host-generated subset rather than an exhaustive set.
`mode` must be `"min"` or `"max"` and returns references tied at the selected
extreme among the values considered by that selector.

The order of references within a selector result is deterministic only for
reproducibility. It carries no graph-theoretic meaning unless a selector's
semantics explicitly state otherwise. Do not use tuple position, adjacency,
prefix/suffix structure, runs, spans, alternation patterns, parity of positions,
or other index-based patterns as graph features.

An extreme selector exposes only the tied references, not the numeric extreme
value. Therefore `mode="max"` does not imply that the maximum is positive or
large. In particular:

- if every sampled witness load for a requested length is zero, a maximum
  witness-load selector may return all considered vertices or edges;
- articulation risk and bridge risk are binary. If no vertex is an articulation
  vertex, `vertices_articulation_risk(mode="max")` may return all vertices. If
  no edge is a bridge, `edges_bridge_risk(mode="max")` may return all edges;
- `non_edges_local_cycle_risk(mode="max")` may return all considered non-edges
  when all have the same common-neighbour count.

Do not assume a positive risk/load merely because a reference came from a
`mode="max"` selector.

# Safe graph API

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
api.matching_k_switch_reconnections_for_edge(edge, k)
api.relocations_legal()
api.relocations_legal_for_edge(edge)
api.edge_fanouts_legal()
api.edge_fanouts_legal_for_edge(edge)
```

## Selector semantics

- `vertices_degree_extreme(mode)` returns all tied vertices having minimum or
  maximum degree in the current private overlay.
- `vertices_degree_class(degree)` returns vertices having exactly that degree in
  the current private overlay.
- `vertices_witness_load_extreme(length, mode)` and
  `edges_witness_load_extreme(length, mode)` accept only a value from
  `ctx.forbidden_lengths`. They rank host-supplied sampled witness loads for that
  cycle length on the current overlay. Missing loads are treated as zero. Only
  tied references are exposed; the numeric witness load is not exposed.
- `vertices_articulation_risk(mode)` ranks vertices by the current binary
  articulation-point indicator: 1 for an articulation vertex and 0 otherwise.
- `edges_bridge_risk(mode)` ranks edges by the current binary bridge indicator:
  1 for a bridge and 0 otherwise.
- `edges_removable()` returns current edge references. The name does NOT mean
  that deletion is guaranteed to preserve connectivity, minimum degree, or final
  graph validity. Those conditions are checked later by `emit()`.
- `vertices_distance_band(source, minimum, maximum)` returns vertices whose
  current-overlay shortest-path distance from the supplied vertex is in the
  inclusive range `[minimum, maximum]`, with `0 <= minimum <= maximum`.
- `non_edges_from_vertex(vertex)` returns absent edges incident to the supplied
  vertex in the current overlay.
- `non_edges_legal()` returns absent-edge references in the current overlay. The
  name does NOT imply that adding one improves or preserves forbidden-cycle
  structure.
- `non_edges_local_cycle_risk(mode)` ranks current absent edges by the number of
  common neighbours of their endpoints. This is a local triangle-closing signal;
  the numeric count is not exposed.
- `paths_length_two()` returns current length-two paths `u-center-v`, represented
  opaquely. It identifies the two incident source edges for a possible fold.
  The selector itself does not reveal `u`, `center`, or `v`.
- `matching_k_switch_reconnections(k)` returns bounded candidate k-switches for
  `k` equal to 2, 3, or 4. A candidate removes `k` pairwise vertex-disjoint
  current edges and adds `k` different edges that reconnect exactly the same
  `2k` endpoints. The candidate set is host-generated and may be a bounded sample,
  not an exhaustive enumeration.
- `matching_k_switch_reconnections_for_edge(edge, k)` has the same semantics, but
  every returned candidate includes the supplied current edge among the removed
  source edges.
- `relocations_legal()` returns endpoint-relocation candidates. For a current
  edge `(u, v)`, a relocation keeps exactly one endpoint and replaces the other
  endpoint by a third vertex `w`, producing either `(u, w)` or `(v, w)` while
  removing `(u, v)`. The replacement edge must currently be absent. The returned
  reference is opaque, so the policy does not learn which endpoint or vertex was
  chosen.
- `relocations_legal_for_edge(edge)` is the same selector restricted to the
  supplied current source edge.
- `edge_fanouts_legal()` returns fanout candidates. For a current edge `(u, v)`
  and a third vertex `w`, fanout removes `(u, v)` and adds both `(u, w)` and
  `(v, w)`. Both added edges must currently be absent. The returned reference is
  opaque.
- `edge_fanouts_legal_for_edge(edge)` is the same selector restricted to the
  supplied current source edge.

Relocation and fanout selectors return syntactically applicable candidates.
They do not guarantee that the final graph remains connected or has minimum
degree at least three. Final graph validity is checked by `emit()`.

# Reproducible choice

Choose reproducibly with:

```python
api.pick(items, seed, salt, feature="uniform")
```

The supplied `seed` must be passed unchanged. `salt` is a printable string or
integer chosen by your policy. `feature` is `"uniform"`, `"degree"`, or
`"inverse_degree"`.

- `"uniform"` chooses uniformly from the supplied opaque references.
- `"degree"` and `"inverse_degree"` are valid only for vertex references and
  weight the choice using current-overlay vertex degree.

Empty input returns `None`. Test that case with `if not selected:`; identity
comparisons such as `selected is None` are outside the accepted AST subset.

# Actions and their exact graph effect

Actions update the current private overlay:

```python
api.add_edge(non_edge)
api.remove_edge(edge)
api.relocate_endpoint(relocation)
api.k_switch(matching)
api.edge_fanout(fanout)
api.edge_fold(path)
```

Their semantics are:

- `add_edge(non_edge)`: add the supplied currently absent edge.
- `remove_edge(edge)`: remove the supplied current edge. Connectivity and minimum
  degree are not guaranteed by the action itself.
- `relocate_endpoint(relocation)`: for the relocation selected above, remove
  `(u, v)` and add either `(u, w)` or `(v, w)`, preserving exactly one endpoint.
- `k_switch(matching)`: remove the candidate's `k` source edges and add its `k`
  reconnection edges on exactly the same endpoint set.
- `edge_fanout(fanout)`: remove `(u, v)` and add `(u, w)` plus `(v, w)`. This
  preserves the degrees of `u` and `v`, increases the degree of `w` by two, and
  increases the edge count by one.
- `edge_fold(path)`: for a selected length-two path `u-center-v`, remove
  `(u, center)` and `(center, v)` and add `(u, v)`. The fold is applicable only
  when `(u, v)` is absent. It preserves the degrees of `u` and `v`, decreases the
  degree of `center` by two, and decreases the edge count by one.

A selector reference can become stale after earlier overlay edits. Prefer simple,
coherent action sequences and obtain selector results from the overlay state on
which they will be used.

# Terminal result and final validation

Finish with exactly one host-minted terminal result:

```python
api.emit()
api.no_plan(reason="EXPLICIT")
```

`emit()` computes the net rewrite relative to the input graph, rejects no-effect
results, requires a connected final graph with minimum degree at least three,
and asks the trusted host to validate the candidate. It returns either a
host-minted `RewritePlan` or `NoPlan("NO_EFFECT" | "ILLEGAL_FINAL_STATE")`.

Allowed explicit no-plan reasons are `"EXPLICIT"`, `"NO_MATCH"`,
`"ILLEGAL_FINAL_STATE"`, and `"NO_EFFECT"`.

The host, not the policy, evaluates the scientific score and decides whether a
valid emitted rewrite improves the incumbent. The policy must not try to infer an
unexposed score from reference order, opaque tokens, or undocumented behavior.

# Python restrictions

- Use only ordinary local variables, conditionals, helper functions, and
  statically bounded loops.
- Helper names must start with `helper_`; use at most 16 helpers.
- The following ordinary Python built-ins are allowed:
  `abs`, `all`, `any`, `bool`, `enumerate`, `int`, `len`, `max`, `min`,
  `range`, `reversed`, `sum`, and `tuple`.
- Do not call any other Python built-in or undocumented function.
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

Return a complete policy with substantive structural selection, a scientifically
motivated mutation strategy, meaningful fallback logic, and exactly one terminal
result on every path. Do not return pseudocode, Markdown, or a declarative
graph-program AST.
