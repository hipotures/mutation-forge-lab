def priority(ctx, proposal):
    """Reviewed structural ranker using only frozen Stage 2B fields."""
    score = sum(proposal["broken_sampled_witnesses_by_length"])
    score -= proposal["local_c4_risk"]
    score += sum(proposal["removed_edge_load_sum_by_length"]) / 1000
    score += sum(proposal["removed_edge_load_max_by_length"]) / 100000
    score += proposal["minimum_distance_between_removed_edges"] / 10000000
    score += proposal["mean_distance_between_removed_edges"] / 100000000
    score += proposal["minimum_preexisting_distance_for_new_edges"] / 1000000000
    score -= proposal["local_triangle_risk"] / 10000000000
    return score
