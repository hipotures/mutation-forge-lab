def priority(ctx, proposal):
    lengths = ctx["forbidden_lengths"]
    counts = ctx["capped_cycle_counts"]
    broken = proposal["broken_sampled_witnesses_by_length"]
    loads = proposal["removed_edge_load_sum_by_length"]
    n = len(lengths)
    stagnation = min(max(ctx["stagnation"], 0), 8)
    remaining = min(max(ctx["remaining_steps"], 0), 64)
    impact = 0
    load_score = 0
    for i in range(16):
        if i < n:
            current = min(max(counts[i], 0), 16)
            hit = min(max(broken[i], 0), 64)
            load = min(max(loads[i], 0), 64)
            impact += (1 + current) * hit
            load_score += (1 + current) * load
    spread = min(max(proposal["mean_distance_between_removed_edges"], 0), 64)
    spread += min(max(proposal["minimum_distance_between_removed_edges"], 0), 64)
    remote = min(max(proposal["mean_preexisting_distance_for_new_edges"], 0), 64)
    remote += min(max(proposal["minimum_preexisting_distance_for_new_edges"], 0), 64)
    triangle = min(max(proposal["local_triangle_risk"], 0), 64)
    c4 = min(max(proposal["local_c4_risk"], 0), 64)
    if stagnation >= 3:
        impact_weight = 6
        load_weight = 3
        spread_weight = 1
        remote_weight = 2
        triangle_weight = 5
        c4_weight = 4
    elif remaining <= 4:
        impact_weight = 7
        load_weight = 3
        spread_weight = 1
        remote_weight = 2
        triangle_weight = 5
        c4_weight = 4
    else:
        impact_weight = 3
        load_weight = 1
        spread_weight = 3
        remote_weight = 3
        triangle_weight = 2
        c4_weight = 2
    score = impact_weight * impact + load_weight * load_score + spread_weight * spread + remote_weight * remote - triangle_weight * triangle - c4_weight * c4
    return score