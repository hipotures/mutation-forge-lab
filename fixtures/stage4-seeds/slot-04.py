def priority(ctx, proposal):
    counts = ctx['capped_cycle_counts']
    broken = proposal['broken_sampled_witnesses_by_length']
    loads = proposal['removed_edge_load_sum_by_length']
    k = proposal['k']
    triangles = proposal['local_triangle_risk']
    c4 = proposal['local_c4_risk']
    coverage = 0
    load_score = 0
    for i in range(16):
        weight = 1
        if i < len(counts):
            weight = 1 + min(counts[i], 8)
        if i < len(broken):
            coverage += broken[i] * weight
        if i < len(loads):
            load_score += loads[i] * weight
    risk = triangles + 2 * c4
    bonus = 0
    if risk <= 6:
        bonus = (k - 2) * 4
    score = coverage * 100 + load_score * 3 - risk * 5 + bonus
    return score