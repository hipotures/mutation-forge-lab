def priority(ctx, proposal):
    broken = proposal["broken_sampled_witnesses_by_length"]
    loads = proposal["removed_edge_load_sum_by_length"]
    load_max = proposal["removed_edge_load_max_by_length"]
    min_removed_distance = proposal["minimum_distance_between_removed_edges"]
    mean_removed_distance = proposal["mean_distance_between_removed_edges"]
    min_new_distance = proposal["minimum_preexisting_distance_for_new_edges"]
    mean_new_distance = proposal["mean_preexisting_distance_for_new_edges"]
    triangle_risk = proposal["local_triangle_risk"]
    c4_risk = proposal["local_c4_risk"]
    tags = proposal["selector_tags"]
    score = 0
    for i in range(16):
        if i < len(broken):
            score += 64 * broken[i] + 4 * loads[i] + load_max[i]
    score += 3 * min_removed_distance
    score += round(2 * mean_removed_distance)
    score += 2 * min_new_distance + round(mean_new_distance)
    score -= 16 * triangle_risk + 8 * c4_risk
    tag_bonus = 0
    for j in range(8):
        if j < len(tags):
            tag = tags[j]
            if tag == "sampled_forbidden_cycle_anchored":
                tag_bonus += 2
            if tag == "high_sampled_witness_load":
                tag_bonus += 2
            if tag == "remote_from_anchor":
                tag_bonus += 1
            if tag == "pairwise_distant_disjoint":
                tag_bonus += 1
            if tag == "mixed_exploit_explore":
                tag_bonus += 1
    return score + min(8, tag_bonus)