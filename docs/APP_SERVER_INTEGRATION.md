# Codex App Server integration boundary

There is no App Server transport or model call in Stage 1. Stage 3 may begin
only after the Stage 2 sandbox report makes an explicit GO decision.

The locally installed `codex-app-server` skill and the installed CLI's
generated experimental schemas are the source of truth. Before implementation,
run the skill's protocol audit without a model turn. Do not hard-code examples
from another CLI release.

The intended use is a thin inference provider, not a coding agent. It must use
a private `CODEX_HOME`, private `CODEX_SQLITE_HOME`, and empty application
working directory. Authentication reuse requires explicit authorization.
Never copy user configuration, project instructions, skills, plugins, hooks,
memories, MCP configuration, or trust settings.

Thread start must provide a short non-empty `baseInstructions` to replace the
built-in coding-agent prompt, empty developer instructions and tools, explicit
capability roots, read-only sandboxing, and no approvals. Initialization,
skill disabling and verification, thread/turn correlation, interleaved event
handling, terminal output, errors, timeouts, process-group cleanup, and final
usage collection must follow the generated protocol.

When persistence is enabled, record the opaque thread path and identifiers.
Record every raw `tokenUsage.last` field and use server-provided
`totalTokens`; do not sum overlapping categories. Inspect saved rollouts for
prompt replacement and absence of project/user capabilities. A small
platform-owned wrapper may remain, so the integration must not claim a bare
model request without rollout evidence.
