"""Write-time fact extraction: one cheap LLM call per session, stored in SQLite.

Facts are short, self-contained statements about the user, each tied to the round it came
from. They make implicit information findable ("my paddle snapped on the lake trip"
-> "The user owns a kayak (outdoor sports gear)") without replacing the original text:
retrieval ranks a round by its user text *or* any of its facts, and the original turn is
what goes into the prompt.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path


EXTRACTOR_VERSION = "v1"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_CHARS_PER_MESSAGE = 6000

PROMPT = """Below are the messages a user sent to an AI assistant in one chat session on {date}.
List every fact this session reveals about the user, so that a later question about their life can be answered from the list alone.

Cover, whenever present (even if mentioned only in passing):
- personal attributes: age, birthday, job, education, location, relationships and family, pets, health conditions
- possessions, devices, gear and purchases (with brand, price, place)
- events and activities, with when they happened; turn relative times ("last Tuesday", "two weeks ago") into absolute dates using the session date
- numbers: counts, amounts, durations, distances, scores
- plans, goals, preferences, opinions and habits
- changes ("I switched to...", "I no longer...")

Rules:
- One fact per line, self-contained: name the things, never "it" or "this".
- Write in the third person ("The user ...").
- When a thing belongs to a broader category a question might use, add it: "The user bought a kayak (outdoor sports gear)".
- Skip generic requests that reveal nothing about the user.
- Answer with JSON only: a list of objects {{"m": <message number>, "fact": "<text>"}}. If there is nothing, answer [].

Messages:
{messages}"""


@dataclass
class Fact:
    round_idx: int
    text: str


class FactStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        # Older fact files named the bookkeeping table "sessions", which clashes with a memory store.
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(sessions)")}
        if "version" in cols:
            self.db.execute("ALTER TABLE sessions RENAME TO fact_runs")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS fact_runs (
                session_id TEXT, version TEXT, model TEXT,
                input_tokens INT, output_tokens INT, raw TEXT,
                PRIMARY KEY (session_id, version));
            CREATE TABLE IF NOT EXISTS facts (
                session_id TEXT, version TEXT, round_idx INT, fact TEXT);
            CREATE INDEX IF NOT EXISTS facts_by_session ON facts(session_id, version);
            """
        )

    def has(self, session_id: str, version: str = EXTRACTOR_VERSION) -> bool:
        with self.lock:
            q = "SELECT 1 FROM fact_runs WHERE session_id = ? AND version = ?"
            return self.db.execute(q, (session_id, version)).fetchone() is not None

    def forget(self, session_id: str) -> None:
        with self.lock, self.db:
            self.db.execute("DELETE FROM facts WHERE session_id = ?", (session_id,))
            self.db.execute("DELETE FROM fact_runs WHERE session_id = ?", (session_id,))

    def get(self, session_id: str, version: str = EXTRACTOR_VERSION) -> list[Fact]:
        with self.lock:
            q = "SELECT round_idx, fact FROM facts WHERE session_id = ? AND version = ? ORDER BY rowid"
            return [Fact(r, f) for r, f in self.db.execute(q, (session_id, version))]

    def put(self, session_id: str, model: str, facts: list[Fact], raw: str, usage: tuple[int, int],
            version: str = EXTRACTOR_VERSION) -> None:
        with self.lock, self.db:
            self.db.execute("DELETE FROM facts WHERE session_id = ? AND version = ?", (session_id, version))
            self.db.executemany(
                "INSERT INTO facts VALUES (?, ?, ?, ?)", [(session_id, version, f.round_idx, f.text) for f in facts]
            )
            self.db.execute(
                "INSERT OR REPLACE INTO fact_runs VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, version, model, usage[0], usage[1], raw),
            )

    def usage(self, version: str = EXTRACTOR_VERSION) -> tuple[int, int, int]:
        with self.lock:
            q = "SELECT COUNT(*), SUM(input_tokens), SUM(output_tokens) FROM fact_runs WHERE version = ?"
            n, i, o = self.db.execute(q, (version,)).fetchone()
            return n, i or 0, o or 0


def _parse(raw: str, n_messages: int) -> list[Fact]:
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    facts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        m, text = item.get("m"), str(item.get("fact", "")).strip()
        if isinstance(m, int) and 1 <= m <= n_messages and text:
            facts.append(Fact(m - 1, text))
    return facts


class Extractor:
    def __init__(self, store: FactStore, model: str = DEFAULT_MODEL, **request: object):
        """`request` adds parameters to each API call, e.g. thinking={"type": "disabled"}."""
        self.store = store
        self.model = model
        self.request = request
        try:
            import anthropic
        except ImportError:
            raise SystemExit("Fact extraction needs the Anthropic SDK: pip install 'pocket-recall[facts]'") from None

        self.client = anthropic.Anthropic(max_retries=6)

    def extract(self, session_id: str, date: str, user_messages: list[str]) -> list[Fact]:
        """user_messages[i] is the user text of round i. Cached per session and extractor version."""
        if self.store.has(session_id):
            return self.store.get(session_id)
        numbered = "\n\n".join(
            f"[{i + 1}] {text[:MAX_CHARS_PER_MESSAGE]}" for i, text in enumerate(user_messages)
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": PROMPT.format(date=date, messages=numbered)}],
            **self.request,
        )
        raw = "".join(b.text for b in response.content if b.type == "text")
        facts = _parse(raw, len(user_messages))
        u = response.usage
        prompt_tokens = u.input_tokens + (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0)
        self.store.put(session_id, response.model, facts, raw, (prompt_tokens, u.output_tokens))
        return facts


def main() -> None:
    """pocket-recall facts DB: extract facts for every session of a store that has none yet."""
    import argparse
    import os

    from tqdm import tqdm

    from .store import MemoryStore

    p = argparse.ArgumentParser(prog="pocket-recall facts", description=main.__doc__)
    p.add_argument("db", type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise SystemExit("Set ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL for a relay) first.")
    store = MemoryStore(args.db.expanduser())
    todo = sum(not store.facts.has(s.session_id) for s in store._sessions())
    bar = tqdm(total=todo, desc="extracting")
    result = store.extract_facts(model=args.model, workers=args.workers, progress=bar.update)
    bar.close()
    print(f"extracted {result['sessions']} sessions, {len(result['failed'])} failed; "
          f"{result['input_tokens']:,} input / {result['output_tokens']:,} output tokens (API-reported)")
    for line in result["failed"][:5]:
        print("  ", line)
    store.close()
