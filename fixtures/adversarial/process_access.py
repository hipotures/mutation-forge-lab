def priority(ctx, proposal):
    return __import__("subprocess").run(["true"]).returncode
