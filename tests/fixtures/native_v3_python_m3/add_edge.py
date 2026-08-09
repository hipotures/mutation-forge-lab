def propose(ctx, graph, api, seed):
    candidates = api.non_edges_legal()
    if not candidates:
        return api.no_plan(reason="NO_MATCH")
    edge = api.pick(candidates, seed, "m3-add-edge")
    if edge == None:  # noqa: E711 - ``is`` is outside the policy AST subset.
        return api.no_plan(reason="NO_MATCH")
    api.add_edge(edge)
    return api.emit()
