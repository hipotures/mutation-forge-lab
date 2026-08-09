def propose(ctx, graph, api, seed):
    candidates = api.edges_removable()
    if not candidates:
        return api.no_plan(reason="NO_MATCH")
    edge = api.pick(candidates, seed, "m3-remove-edge")
    if edge == None:  # noqa: E711 - ``is`` is outside the policy AST subset.
        return api.no_plan(reason="NO_MATCH")
    api.remove_edge(edge)
    return api.emit()
