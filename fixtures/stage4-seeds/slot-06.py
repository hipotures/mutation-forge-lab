def priority(ctx, proposal):
    counts = ctx["capped_cycle_counts"]
    broken = proposal["broken_sampled_witnesses_by_length"]
    witness = 0.0
    for i in range(16):
        if i < len(counts) and i < len(broken):
            x = min(1000000, broken[i] * (1 + counts[i]))
            witness = witness + x / (1 + x)
    witness = witness / len(counts)
    spread = min(1000000, proposal["mean_distance_between_removed_edges"])
    remote = min(1000000, proposal["minimum_preexisting_distance_for_new_edges"])
    risk = min(1000000, proposal["local_triangle_risk"] + proposal["local_c4_risk"])
    return 0.45 * witness + 0.30 * spread / (1 + spread) + 0.20 * remote / (1 + remote) - 0.05 * risk / (1 + risk)