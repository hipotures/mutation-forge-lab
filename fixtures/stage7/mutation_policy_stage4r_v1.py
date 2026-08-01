def priority(ctx, proposal):
    counts = ctx["capped_cycle_counts"]
    broken = proposal["broken_sampled_witnesses_by_length"]
    loads = proposal["removed_edge_load_sum_by_length"]
    n = min(16, max(1, len(counts)))
    coverage = 0.0
    load = 0.0
    for i in range(16):
        if i < n:
            c = min(1000000, max(0, counts[i]))
            b = min(1000000, max(0, broken[i]))
            l = min(1000000, max(0, loads[i]))
            weight = 1.0 + c / (1.0 + c)
            coverage = coverage + weight * b / (1.0 + b)
            load = load + weight * l / (1.0 + l)
    coverage = coverage / (2.0 * n)
    load = load / (2.0 * n)
    triangle = min(1000000, max(0, proposal["local_triangle_risk"]))
    c4 = min(1000000, max(0, proposal["local_c4_risk"]))
    risk = (triangle + c4) / (1.0 + triangle + c4)
    return coverage + 0.15 * load - 0.02 * risk