"""Claude Code hook: save each session into the memory store as it goes.

Configured as an async Stop hook, it gets {"transcript_path": ...} on stdin after every turn
and adds the turn's new messages (your prompts and the assistant's written replies) to
$POCKET_RECALL_DB (default ~/.pocket-recall/memory.db). It never fails the session: problems go to
~/.pocket-recall/hook.log and the hook exits 0.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

HOME = Path("~/.pocket-recall").expanduser()

try:
    import fcntl
except ImportError:  # Windows: no flock; concurrent saves are rare and SQLite serialises writes
    fcntl = None


def save_transcript(transcript: Path, db: Path) -> int:
    from .importers import _add_new, read_claude_code_session
    from .store import MemoryStore

    messages = read_claude_code_session(transcript)
    if not any(m["role"] == "user" for m in messages):
        return 0
    store = MemoryStore(db)
    try:
        return _add_new(store, "cc-" + transcript.stem[:12], messages)  # same id as pocket-recall import
    finally:
        store.close()


def main() -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    log = HOME / "hook.log"
    try:
        payload = json.load(sys.stdin)
        transcript = Path(payload.get("transcript_path") or "")
        if not transcript.is_file():
            return
        db = Path(os.environ.get("POCKET_RECALL_DB", HOME / "memory.db")).expanduser()
        with open(HOME / ".hook.lock", "w") as lock:
            try:
                if fcntl:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return  # another save is running; the next turn's hook catches up
            added = save_transcript(transcript, db)
        if added:
            with log.open("a") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} saved {added} messages from {transcript.name}\n")
    except Exception:
        with log.open("a") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} error\n{traceback.format_exc()}\n")
    finally:
        sys.exit(0)
