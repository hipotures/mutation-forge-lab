state = 0


def priority(ctx, proposal):
    global state
    state += 1
    return state
