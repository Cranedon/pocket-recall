"""Automatic memory injection: a Claude Code UserPromptSubmit hook.

Before Claude sees a prompt, the hook asks the running pocket-recall MCP server (over a Unix
socket in ~/.pocket-recall/run/, so the embedding model is already loaded) for the memory most
relevant to the prompt, leaving out the current session. If the relevance gate passes, the
result is added to Claude's context; otherwise nothing is added and Claude can still call the
`recall` tool itself. Within one session a message is injected at most once, and when the best
match brings nothing new nothing is added, so follow-ups on one topic do not repeat the same
history. The hook never blocks a prompt: any problem means no injection.

The gate (`passes`) was recalibrated on two weeks of real prompts: most of them are follow-ups
inside a long session, whose context is already in the conversation, and messages that tools
type into the prompt (task notifications, "usage limit reset"). Those are skipped, and memory is
injected only when it matches better than the session's own recent prompts and, past the first
few prompts of a session, shares rare words with the prompt.

Only the standard library is imported here, so starting the hook stays fast.

    POCKET_RECALL_INJECT=0              turn injection off
    POCKET_RECALL_INJECT_BUDGET=1000    tokens of memory to add at most
    POCKET_RECALL_INJECT_THRESHOLD=0.6  relevance needed to inject (see core.relevance)
"""

from __future__ import annotations

import json
import os
import re
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
# Chosen on replayed real prompts (one half of them; docs/EVALUATION.md, section 12).
OWN_MARGIN = 0.1  # memory must beat the best match among the session's own recent prompts by this much
EARLY_PROMPTS = 3  # within a session's first prompts that is enough; later, the prompt must also
LATE_RARE_HITS = 2  # share this many rare words with the best memory keys
RECENT_PROMPTS = 30
# Recent prompts are read from the end of the transcript, widening once: sessions full of tool
# output and images can hold only a few prompts in their last megabytes.
TAIL_READS = (2_000_000, 16_000_000)
MAX_PROMPT_LINE = 200_000  # longer transcript lines are tool output, skipped without parsing
# Text that Claude Code or the user's tools type into the prompt; it is not a question to look up.
SYSTEM_PROMPT = re.compile(
    r"^\s*<(?!pasted_content)[a-z][a-z0-9_-]*(\s[^>]*)?/?>|"  # <task-notification>, <bash-input>, ... (not a paste)
    r"^\s*(I hit my usage limit|在您工作期间，我达到了使用限额|This session is being continued|"
    r"\[Request interrupted|Caveat: )")


def is_system_prompt(prompt: str) -> bool:
    return bool(SYSTEM_PROMPT.search(prompt))


def recent_prompts(transcript_path: str, limit: int = RECENT_PROMPTS) -> tuple[list[str], bool]:
    """The user's last prompts in this session, oldest first, read from the end of its transcript,
    and whether the whole transcript was read (if not, the session has more prompts than found)."""
    out: list[str] = []
    whole = True
    for cap in TAIL_READS:
        try:
            with open(transcript_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - cap))
                lines = f.read().decode("utf-8", "ignore").splitlines()[1 if size > cap else 0:]
        except OSError:
            return [], True
        whole = size <= cap
        out = []
        for line in lines:
            if len(line) > MAX_PROMPT_LINE or '"user"' not in line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            msg = e.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if e.get("type") != "user" or e.get("isMeta") or e.get("isSidechain") or not isinstance(content, str):
                continue
            content = content.strip()
            if len(content) >= MIN_PROMPT_CHARS and not content.startswith("/") and not is_system_prompt(content):
                out.append(content)
        if whole or len(out) >= limit:
            break
    return out[-limit:], whole

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


def passes(result: dict, threshold: float, recent: int = 0) -> bool:
    """A question about a period that has records ("what did I do last week"), or memory that is
    relevant (`threshold`), matches better than the session's own recent prompts and, once the
    session has `recent` >= EARLY_PROMPTS prompts, shares rare words with the prompt."""
    if not result or not result.get("text"):
        return False
    if result.get("time_only") and result.get("window_records"):
        return True
    if (result.get("relevance") or 0) < threshold:
        return False
    if "own" not in result:
        return True  # a server from before 0.3 sends only relevance: its gate
    if result["cos"] <= result["own"] + OWN_MARGIN:
        return False
    return recent < EARLY_PROMPTS or result["rare_hits"] >= LATE_RARE_HITS


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
        if len(prompt) < MIN_PROMPT_CHARS or prompt.startswith("/") or is_system_prompt(prompt):
            return
        transcript = payload.get("transcript_path") or ""
        session = session_id_for(transcript) if transcript else ""
        seen = _load_seen(session) if session else []
        recent, whole = recent_prompts(transcript) if transcript else ([], True)
        if recent and recent[-1] == prompt:
            recent = recent[:-1]  # the prompt itself, if the transcript already has it
        # a long transcript read only in part is not a new session, however few prompts were found
        n_recent = len(recent) if whole else max(len(recent), EARLY_PROMPTS)
        result = ask({
            "query": prompt,
            "budget": int(os.environ.get("POCKET_RECALL_INJECT_BUDGET", DEFAULT_BUDGET)),
            "exclude": [session] if session else [],
            "seen": seen,
            "context": recent,
        })
        threshold = float(os.environ.get("POCKET_RECALL_INJECT_THRESHOLD", DEFAULT_THRESHOLD))
        if not passes(result, threshold, n_recent):
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
