def priority(ctx, proposal):
    """Reviewed Stage 2A weighted probe ranker."""
    score = proposal["features"]["weight"] - proposal["features"]["penalty"]
    return score
