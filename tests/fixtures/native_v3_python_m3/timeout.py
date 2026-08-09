def propose(ctx, graph, api, seed):
    total = 0
    for first in range(64):
        for second in range(64):
            for third in range(64):
                for fourth in range(64):
                    total += first + second + third + fourth
    if total < 0:
        return api.no_plan(reason="NO_MATCH")
    return api.no_plan()
