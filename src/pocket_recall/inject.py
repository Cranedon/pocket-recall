"""Automatic memory injection: a Claude Code UserPromptSubmit hook.

Before Claude sees a prompt, the hook asks the running pocket-recall MCP server (over a Unix
socket in ~/.pocket-recall/run/, so the embedding model is already loaded) for the memory most
relevant to the prompt, leaving out the current session. If the relevance gate passes, the
result is added to Claude's context; otherwise nothing is added and Claude can still call the
`recall` tool itself. Within one session a message is injected at most once, and when the best
match brings nothing new nothing is added, so follow-ups on one topic do not repeat the same
history. The hook never blocks a prompt: any problem means no injection.

Only the standard library is imported here, so starting the hook stays fast.

    POCKET_RECALL_INJECT=0              turn injection off
    POCKET_RECALL_INJECT_BUDGET=1000    tokens of memory to add at most
    POCKET_RECALL_INJECT_THRESHOLD=0.6  relevance needed to inject (see core.relevance)
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

RUN_DIR = Path("~/.pocket-recall/run").expanduser()
DEFAULT_BUDGET = 1000
# Chosen on development data. Fires on 72% (LongMemEval) / 97% (LoCoMo dev) of questions whose
# answer is in memory and on 13% / 10% of questions asked against someone else's history; 0.65
# would cut those to 7% / 3% but miss a third of real users' questions about their own past.
DEFAULT_THRESHOLD = 0.6
MIN_PROMPT_CHARS = 8  # "ok", "继续", "yes" carry no question to look up
TIMEOUT_S = 4.0


def session_id_for(transcript_path: str) -> str:
    """The id the Stop hook and the importer give this transcript's session."""
    return "cc-" + Path(transcript_path).stem[:12]


def _servers() -> list[Path]:
    """Sockets of live MCP servers, newest first; stale ones are removed."""
    socks = sorted(RUN_DIR.glob("mcp-*.sock"), key=lambda p: p.stat().st_mtime, reverse=True) if RUN_DIR.exists() else []
    alive = []
    for path in socks:
        try:
            os.kill(int(path.stem.split("-")[1]), 0)
            alive.append(path)
        except (ProcessLookupError, ValueError):
            path.unlink(missing_ok=True)
        except PermissionError:
            alive.append(path)
    return alive


def ask(request: dict, timeout: float = TIMEOUT_S) -> dict | None:
    for path in _servers():
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect(str(path))
                s.sendall(json.dumps(request).encode() + b"\n")
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
            return json.loads(buf)
        except (OSError, ValueError):
            continue
    return None


def _seen_file(session: str) -> Path:
    return RUN_DIR / f"seen-{session}.json"


def _load_seen(session: str) -> list[int]:
    try:
        return json.loads(_seen_file(session).read_text())
    except (OSError, ValueError):
        return []


def _save_seen(session: str, ids: list[int]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _seen_file(session).write_text(json.dumps(ids))
    for old in RUN_DIR.glob("seen-*.json"):  # sessions idle for a week
        if old.stat().st_mtime < time.time() - 7 * 86400:
            old.unlink(missing_ok=True)


def passes(result: dict, threshold: float) -> bool:
    """Relevant enough, or a question about a period that has records ("what did I do last week")."""
    if not result or not result.get("text"):
        return False
    return (result.get("relevance") or 0) >= threshold or bool(result.get("time_only") and result.get("window_records"))


def context_for(result: dict) -> str:
    return (
        f"pocket-recall: notes from earlier sessions that may bear on this prompt "
        f"(relevance {result.get('relevance') or 0:.2f}, verbatim, oldest first). Use them only if they help; "
        f"the `recall` tool can search the full history.\n\n{result['text']}"
    )


def main() -> None:
    try:
        if os.environ.get("POCKET_RECALL_INJECT", "1") == "0":
            return
        payload = json.load(sys.stdin)
        prompt = (payload.get("prompt") or "").strip()
        if len(prompt) < MIN_PROMPT_CHARS or prompt.startswith("/"):
            return
        session = session_id_for(payload["transcript_path"]) if payload.get("transcript_path") else ""
        seen = _load_seen(session) if session else []
        result = ask({
            "query": prompt,
            "budget": int(os.environ.get("POCKET_RECALL_INJECT_BUDGET", DEFAULT_BUDGET)),
            "exclude": [session] if session else [],
            "seen": seen,
        })
        if not passes(result, float(os.environ.get("POCKET_RECALL_INJECT_THRESHOLD", DEFAULT_THRESHOLD))):
            return
        if session:
            _save_seen(session, seen + result.get("ids", []))
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context_for(result),
        }}, ensure_ascii=False))
    except Exception:
        pass  # injection is a bonus: never block or break the prompt
    finally:
        sys.exit(0)
