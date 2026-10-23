---
name: import-history
description: Import the user's older chats into pocket-recall memory, mask secrets in an existing store, or open the local memory browser. Use when the user asks to import past Claude Code sessions or a claude.ai export, to see or search what is stored, or to clean API keys out of the memory.
---

# Manage the pocket-recall memory

All commands run the plugin's own copy of pocket-recall. Run them with the Bash tool exactly as written; the paths are filled in by Claude Code.

```bash
PR() { UV_PROJECT_ENVIRONMENT="${CLAUDE_PLUGIN_DATA}/venv" uv run --quiet --frozen --no-dev --project "${CLAUDE_PLUGIN_ROOT}" pocket-recall "$@"; }
```

Define `PR` in the same command as its use, since each Bash call starts a fresh shell.

## Import past Claude Code sessions

The hook only saves sessions from the moment the plugin was installed. To add everything before that:

```bash
PR() { UV_PROJECT_ENVIRONMENT="${CLAUDE_PLUGIN_DATA}/venv" uv run --quiet --frozen --no-dev --project "${CLAUDE_PLUGIN_ROOT}" pocket-recall "$@"; }; PR import claude-code ~/.pocket-recall/memory.db
```

Add `--project NAME` to import a single project. Re-running only adds what is new. Only the user's prompts and the assistant's written replies are kept: tool calls, tool output, thinking and sub-agents are skipped. Report the numbers it prints.

## Import a claude.ai export

The user downloads it from claude.ai: Settings → Privacy → Export data. Then, with the zip path they give you:

```bash
PR() { UV_PROJECT_ENVIRONMENT="${CLAUDE_PLUGIN_DATA}/venv" uv run --quiet --frozen --no-dev --project "${CLAUDE_PLUGIN_ROOT}" pocket-recall "$@"; }; PR import claude ~/.pocket-recall/memory.db /path/to/export.zip
```

## Browse the memory

```bash
PR() { UV_PROJECT_ENVIRONMENT="${CLAUDE_PLUGIN_DATA}/venv" uv run --quiet --frozen --no-dev --project "${CLAUDE_PLUGIN_ROOT}" pocket-recall "$@"; }; PR web --dir ~/.pocket-recall
```

Run it in the background and tell the user to open http://localhost:8765. The page lists every session, searches the original text, and shows exactly what `recall` would return for a question at a chosen token budget.

## Mask secrets

New messages are masked on write (API keys, tokens, private keys, passwords in URLs and assignments become `[REDACTED:kind]`). For a store written before that, or after the user turned it off:

```bash
PR() { UV_PROJECT_ENVIRONMENT="${CLAUDE_PLUGIN_DATA}/venv" uv run --quiet --frozen --no-dev --project "${CLAUDE_PLUGIN_ROOT}" pocket-recall "$@"; }; PR redact ~/.pocket-recall/memory.db
```

## Forget

To delete a session, use the `forget_session` tool of the MCP server with the session id shown in the web page. Deleting is permanent: confirm with the user first.
