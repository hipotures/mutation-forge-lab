# Mutation Forge Lab contributor instructions

- Stage 1 is a deterministic baseline harness. Do not add generated-code
  execution, LLM calls, evolutionary search, or production HEG integration.
- Treat the sibling `../heg` repository as read-only. Record its exact commit
  and dirty state in run metadata.
- Use Python 3.12 or newer and `uv`; never invoke `pip` directly.
- Keep source, tests, configuration, prompts, schemas, logs, and documentation
  in English.
- Preserve deterministic seeds, equal evaluation budgets, and immutable
  dataset manifests when comparing policies.
- A heuristic score of zero is only a submission to exact verification. It is
  never a verified counterexample by itself.
- Generated policy code, sandboxing, k-switch proposals, and Codex App Server
  transport belong to later milestones and must not be implemented in Stage 1.
- Do not pipe monitored command output through `head`, `tail`, `less`, or
  `more`.
- Snapshot SQLite databases with SQLite's online backup API or `.backup`;
  never copy a live database with plain `cp`.
- Run relevant tests and static checks before committing. Commit only changes
  made for the current task.
