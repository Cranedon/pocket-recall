"""Import your own chat history into a memory store.

    pocket-recall import claude-code DB [--project NAME]   # Claude Code sessions in ~/.claude/projects
    pocket-recall import claude DB EXPORT                  # claude.ai export (zip or conversations.json)

Only what you typed and the assistant's written replies are kept: tool calls, tool output,
thinking, system notices and sub-agent runs are dropped. Original timestamps are kept, and
re-running an import only adds what is new.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

from .store import MemoryStore, normalize_time

_SYSTEM_TAGS = re.compile(r"<(system-reminder|local-command-[a-z]+|command-[a-z]+)>.*?</\1>", re.S)
_NOTICE_START = ("<task-notification", "<local-command", "<command-", "<system-reminder", "Caveat:")


def _clean(text: str) -> str:
    return _SYSTEM_TAGS.sub("", text).strip()


def _text_blocks(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


# --- Claude Code ---------------------------------------------------------------------------


def _is_human(line: dict) -> bool:
    if line.get("type") != "user" or line.get("isMeta") or line.get("isSidechain"):
        return False
    if line.get("isCompactSummary") or line.get("isVisibleInTranscriptOnly"):
        return False  # the summary Claude Code writes when it compacts a long session
    content = line.get("message", {}).get("content")
    if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
        return False
    origin = line.get("origin")
    if isinstance(origin, dict) and origin.get("kind"):
        return origin["kind"] == "human"
    return not _text_blocks(content).lstrip().startswith(_NOTICE_START)  # older files have no origin


def read_claude_code_session(path: Path) -> list[dict]:
    """Messages of one Claude Code transcript: your prompts and the assistant's written replies."""
    messages: list[dict] = []
    for raw in path.open(encoding="utf-8"):
        try:
            line = json.loads(raw)
        except json.JSONDecodeError:
            continue
        when = line.get("timestamp")
        if _is_human(line):
            text = _clean(_text_blocks(line["message"].get("content")))
            if text:
                messages.append({"role": "user", "content": text, "created_at": when})
        elif line.get("type") == "assistant" and not line.get("isSidechain") and messages:
            text = _text_blocks(line.get("message", {}).get("content")).strip()
            if not text:
                continue
            if messages[-1]["role"] == "assistant":  # one turn spans several API responses
                messages[-1]["content"] += "\n\n" + text
            else:
                messages.append({"role": "assistant", "content": text, "created_at": when})
    return messages


def import_claude_code(store: MemoryStore, projects_dir: Path, project: str | None = None) -> dict:
    files = sorted(projects_dir.expanduser().glob(f"{project or '*'}/*.jsonl"))  # top level only: no sub-agents
    added_sessions = added_messages = 0
    for f in files:
        messages = read_claude_code_session(f)
        if not any(m["role"] == "user" for m in messages):
            continue
        added_messages += _add_new(store, "cc-" + f.stem[:12], messages)
        added_sessions += 1
    return {"files": len(files), "sessions": added_sessions, "messages": added_messages}


# --- claude.ai export ------------------------------------------------------------------------


def _load_export(path: Path) -> list[dict]:
    path = path.expanduser()
    if path.is_dir():
        path = path / "conversations.json"
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("conversations.json"))
            return json.loads(z.read(name))
    return json.loads(path.read_text(encoding="utf-8"))


def read_claude_export(path: Path) -> list[tuple[str, str, list[dict]]]:
    """(conversation id, created_at, messages) for each conversation in a claude.ai export."""
    out = []
    for conv in _load_export(path):
        messages = []
        for m in conv.get("chat_messages", []):
            text = _text_blocks(m.get("content")) or m.get("text") or ""
            for att in m.get("attachments") or []:
                if att.get("extracted_content"):
                    text += f"\n\n[附件 {att.get('file_name', '')}]\n{att['extracted_content']}"
            role = {"human": "user", "assistant": "assistant"}.get(m.get("sender"))
            if role and text.strip():
                messages.append({"role": role, "content": text.strip(), "created_at": m.get("created_at")})
        if messages:
            out.append((conv.get("uuid", ""), conv.get("created_at") or messages[0]["created_at"], messages))
    return out


def import_claude_export(store: MemoryStore, path: Path) -> dict:
    added_sessions = added_messages = 0
    for conv_id, _, messages in read_claude_export(path):
        n = _add_new(store, "claude-" + conv_id[:12], messages)
        added_messages += n
        added_sessions += n > 0
    return {"sessions": added_sessions, "messages": added_messages}


# --- shared ----------------------------------------------------------------------------------


def _add_new(store: MemoryStore, session_id: str, messages: list[dict]) -> int:
    """Store messages of a session, skipping those already imported (by timestamp)."""
    existing = store.get_session(session_id)
    last = max((m.created_at for m in existing.messages), default="") if existing else ""
    new = [{**m, "created_at": normalize_time(m["created_at"])} for m in messages]
    new = [m for m in new if m["created_at"] > last]
    if not new:
        return 0
    if existing:
        for m in new:
            store.add_message(m["role"], m["content"], session_id=session_id, created_at=m["created_at"])
    else:
        store.add_session(new, session_id=session_id, started_at=new[0]["created_at"])
    return len(new)


def main() -> None:
    p = argparse.ArgumentParser(prog="pocket-recall import", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="source", required=True)
    sp = sub.add_parser("claude-code", help="Claude Code sessions")
    sp.add_argument("db", type=Path)
    sp.add_argument("--projects-dir", type=Path, default=Path("~/.claude/projects"))
    sp.add_argument("--project", help="one project folder name, e.g. -Users-me-Projects-foo")
    sp = sub.add_parser("claude", help="claude.ai data export")
    sp.add_argument("db", type=Path)
    sp.add_argument("export", type=Path, help="the export zip, its folder, or conversations.json")
    args = p.parse_args()
    store = MemoryStore(args.db.expanduser())
    if args.source == "claude-code":
        r = import_claude_code(store, args.projects_dir, args.project)
        print(f"read {r['files']} transcripts; {r['messages']} new messages in {r['sessions']} sessions")
    else:
        r = import_claude_export(store, args.export)
        print(f"{r['messages']} new messages in {r['sessions']} conversations")
    s = store.stats()
    print(f"{args.db}: {s['sessions']} sessions, {s['messages']} messages (~{s['tokens_est']:,} tokens)")
    store.close()
