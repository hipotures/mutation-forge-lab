def priority(ctx, proposal):
    """Reviewed Stage 2A bounded-loop probe ranker."""
    total = 0
    for value in proposal["features"]["values"]:
        total += value
    return total
