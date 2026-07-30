def priority(ctx, proposal):
    lengths = ctx["forbidden_lengths"]
    counts = ctx["capped_cycle_counts"]
    order = ctx["order"]
    broken = proposal["broken_sampled_witnesses_by_length"]
    loads = proposal["removed_edge_load_sum_by_length"]
    separation = proposal["mean_distance_between_removed_edges"]
    n = len(lengths)
    hit_total = 0.0
    load_total = 0.0
    for i in range(16):
        if i < n:
            count_scale = max(1, counts[i])
            hit_total = hit_total + broken[i] / count_scale
            load_total = load_total + loads[i] / max(1, count_scale * 4)
    hit_fraction = hit_total / max(1, n)
    load_fraction = load_total / max(1, n)
    normalized_separation = separation / max(1, order)
    score = 5.0 * hit_fraction + 2.0 * load_fraction + normalized_separation
    return score