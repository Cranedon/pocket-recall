"""MCP server: gives any MCP-capable assistant a long-term memory backed by one SQLite file.

Run with `pocket-recall mcp` (stdio). The store lives at $POCKET_RECALL_DB, default ~/.pocket-recall/memory.db.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime

from mcp.server.mcpserver import MCPServer

from . import __version__
from .store import MemoryStore, normalize_time

INSTRUCTIONS = """Long-term memory of past conversations with this user.
- At the start of a task that may depend on the user's past (their preferences, history, plans,
  things they told you before), call `recall` with a question describing what you need."""
SAVE_MANUALLY = """
- When a conversation ends, or the user shares something worth remembering, call
  `save_conversation` with the messages exchanged, verbatim. Nothing is summarised: store
  what was actually said."""
SAVED_AUTOMATICALLY = """
- This client saves every conversation automatically. Do not call `save_conversation` for the
  current conversation; use it only to import older chats the user gives you."""
# The Claude Code plugin sets POCKET_RECALL_AUTOSAVE=1 because its Stop hook already stores each turn.
AUTOSAVE = os.environ.get("POCKET_RECALL_AUTOSAVE") == "1"
INSTRUCTIONS += SAVED_AUTOMATICALLY if AUTOSAVE else SAVE_MANUALLY

server = MCPServer("pocket-recall", instructions=INSTRUCTIONS, version=__version__,
                   website_url="https://github.com/Cranedon/pocket-recall")
_store: MemoryStore | None = None


def store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore(os.environ.get("POCKET_RECALL_DB", "~/.pocket-recall/memory.db"))
    return _store


@server.tool()
def recall(query: str, budget_tokens: int = 8000) -> str:
    """Find what the user said in past conversations that is relevant to `query`.

    Returns verbatim excerpts, oldest first, each under its session date, fitting in
    `budget_tokens`. Use the current date to resolve relative times ("last week").
    """
    result = store().recall(query, budget_tokens)
    if not result.sessions:
        return "No stored conversations yet."
    header = f"Current date: {datetime.now().astimezone():%Y-%m-%d %H:%M}. "
    if result.window and not result.window_records:
        header += (f"There are no stored conversations from {result.window[0]} to {result.window[1]}, the period "
                   "the question refers to; the excerpts below were matched by content instead. ")
    elif result.window:
        header += f"The question refers to {result.window[0]} to {result.window[1]}; sessions from then rank higher. "
    header += f"{len(result.sessions)} past sessions matched; excerpts follow.\n\n"
    return header + result.text


@server.tool()
def save_conversation(messages: list[dict[str, str]], session_id: str | None = None,
                      started_at: str | None = None) -> str:
    """Store messages verbatim: a list of {"role": "user" | "assistant", "content": "..."}.

    Pass the same `session_id` again to append to an earlier session; omit it to start a new one.
    `started_at` is when the conversation happened, e.g. "2025-03-14" or "2025-03-14T09:30";
    omit it for a conversation happening now, but always give it when saving an older one,
    because recall uses dates to answer questions like "what did I do last week".
    A message may also carry its own "created_at".
    """
    clean = [m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]
    if not clean:
        return "Nothing to save: messages need a role of 'user' or 'assistant' and non-empty content."
    try:  # check every date before writing anything
        times = [normalize_time(m.get("created_at") or started_at) for m in clean]
    except ValueError as e:
        return f"Not saved: could not read a date ({e}). Use a form like 2025-03-14 or 2025-03-14T09:30."
    if session_id:
        for m, at in zip(clean, times):
            store().add_message(m["role"], m["content"], session_id=session_id, created_at=at)
    else:
        session_id = store().add_session(
            [{**m, "created_at": at} for m, at in zip(clean, times)], started_at=times[0]
        )
    return f"Saved {len(clean)} messages to session {session_id}."


@server.tool()
def forget_session(session_id: str) -> str:
    """Permanently delete one stored session and all its messages."""
    return f"Deleted session {session_id}." if store().forget_session(session_id) else f"No session {session_id}."


@server.tool()
def memory_stats() -> str:
    """How much is stored and where."""
    s = store().stats()
    return f"{s['sessions']} sessions, {s['messages']} messages (~{s['tokens_est']:,} tokens) in {s['path']}"


def _warm_up() -> None:
    """Load (on first run: download) the embedding model in the background, so the first
    recall does not wait for it."""
    try:
        embedder = store()._embedder
        if embedder:
            embedder.query("warm up")
    except Exception:
        pass  # recall will raise the real error when it is called


def main() -> None:
    threading.Thread(target=_warm_up, daemon=True).start()
    server.run("stdio")
