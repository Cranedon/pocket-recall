"""A personal memory store in one SQLite file.

Nothing is summarised away: every message is kept as written. `recall` picks what fits a
token budget for a given question, using the same retrieval the benchmark measures
(`pocket_recall.core.select`). Embeddings are computed locally and cached in the same file.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from .core import REPLY_LANE_SHARE, estimate_tokens, relevance, select, split_rounds
from .embed import Embedder, Reranker
from .facts import DEFAULT_MODEL as FACTS_MODEL, Extractor, FactStore
from .redact import enabled as redact_enabled, redact
from .timeparse import is_time_only, time_reference

# Multilingual by default: on LongMemEval (English) it is within 1-2 points of bge-small-en's
# evidence recall, and on a small Chinese check it finds 18/20 where bge-small-en finds 5/20.
# A store's vectors belong to one model; switching re-embeds on the next recall.
DEFAULT_EMBED_MODEL = os.environ.get("POCKET_RECALL_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

DEFAULT_RERANK_MODEL = os.environ.get("POCKET_RECALL_RERANK_MODEL") or None  # measured below before turning on

# A whole round (user turn plus the assistant's reply) costs several hundred tokens. With a small
# budget, three of them would leave little room for anything else.
FULL_ROUND_TOKENS = 2500

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_session ON messages(session_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_time(value: str | None) -> str:
    """Local-time ISO 8601 from "2024-03-05", "2024/03/05 14:20", ISO strings (any offset), or None (now).
    Times without an offset are taken as local time. Raises ValueError for anything else."""
    if not value:
        return _now()
    text = value.strip().replace("/", "-").replace("Z", "+00:00")
    if len(text) > 10 and text[10] == " ":
        text = text[:10] + "T" + text[11:]
    # Stored in local time: date ranges ("last week") are compared by local calendar day, and a
    # UTC timestamp from late evening would otherwise land on the next day.
    return datetime.fromisoformat(text).astimezone().isoformat(timespec="seconds")


@dataclass
class Message:
    id: int
    session_id: str
    role: str
    content: str
    created_at: str


@dataclass
class RecalledSession:
    session_id: str
    started_at: str
    messages: list[Message]
    rank: int = 0  # 1 = the session whose round was picked first, i.e. the most relevant


@dataclass
class Recall:
    query: str
    sessions: list[RecalledSession] = field(default_factory=list)
    rounds_considered: int = 0
    window: tuple[date, date] | None = None  # the period the question refers to, if any
    time_only: bool = False  # the question named only a time, so that period came strictly first
    window_records: int = 0  # rounds dated inside `window`; 0 means nothing happened then
    relevance: float | None = None  # gate score (core.relevance); None without embeddings

    @property
    def text(self) -> str:
        """History rendered for a prompt, oldest session first."""
        parts = []
        for s in self.sessions:
            day = s.started_at[:10]
            body = "\n\n".join(
                f"{m.role}{'' if m.created_at[:10] == day else f' ({m.created_at[:10]})'}: {m.content.strip()}"
                for m in s.messages
            )
            parts.append(f"### Session {s.started_at[:16].replace('T', ' ')}\n{body}")
        return "\n\n".join(parts)

    @property
    def tokens_est(self) -> int:
        return estimate_tokens(self.text)


class MemoryStore:
    def __init__(self, path: str | Path, *, scorer: str = "hybrid", full_rounds: int | None = None,
                 embed_model: str = DEFAULT_EMBED_MODEL, max_turn_tokens: int | None = 300,
                 rerank_model: str | None = DEFAULT_RERANK_MODEL,
                 assistant_keys: bool = os.environ.get("POCKET_RECALL_ASSISTANT_KEYS", "") == "1",
                 redact_secrets: bool | None = None,
                 reply_lane: float | None = None):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        self.scorer = scorer
        self.full_rounds = full_rounds  # None: one per FULL_ROUND_TOKENS of budget, at most 3
        self.max_turn_tokens = max_turn_tokens
        self.assistant_keys = assistant_keys
        self.redact_secrets = redact_enabled() if redact_secrets is None else redact_secrets
        self.reply_lane = float(os.environ.get("POCKET_RECALL_REPLY_LANE", str(REPLY_LANE_SHARE))) if reply_lane is None else reply_lane
        self._embedder = Embedder(self.path, embed_model) if scorer != "bm25" else None
        self._reranker = Reranker(rerank_model) if rerank_model and scorer != "bm25" else None
        self.facts = FactStore(self.path)  # optional write-time facts, same file

    def close(self) -> None:
        self.db.close()
        self.facts.db.close()
        if self._embedder:
            self._embedder.db.close()

    # --- writing -------------------------------------------------------------------------

    def add_message(self, role: str, content: str, *, session_id: str | None = None,
                    created_at: str | None = None) -> str:
        """Append one message; starts a new session when `session_id` is new or omitted."""
        session_id = session_id or uuid.uuid4().hex[:12]
        created_at = normalize_time(created_at)
        content = redact(content) if self.redact_secrets else content
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO sessions (id, started_at) VALUES (?, ?)", (session_id, created_at)
            )
            self.db.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, created_at),
            )
        if role == "user" and self._embedder:
            self._embedder.passages([content])  # embed now so recall stays fast
        return session_id

    def add_session(self, messages: Iterable[dict], *, session_id: str | None = None,
                    started_at: str | None = None) -> str:
        """Store a whole conversation: an iterable of {"role", "content"[, "created_at"]} dicts.

        Pass the original `started_at` when importing old chats; dates drive time-aware recall.
        """
        session_id = session_id or uuid.uuid4().hex[:12]
        started_at = normalize_time(started_at)
        rows = [
            (m["role"], redact(m["content"]) if self.redact_secrets else m["content"],
             normalize_time(m["created_at"]) if m.get("created_at") else started_at)
            for m in messages if m.get("content")
        ]
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO sessions (id, started_at) VALUES (?, ?)", (session_id, started_at)
            )
            self.db.executemany(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                [(session_id, role, content, at) for role, content, at in rows],
            )
        if self._embedder:
            self._embedder.passages([c for r, c, _ in rows if r == "user"])
        return session_id

    def redact_existing(self) -> int:
        """Mask secrets in messages stored before redaction was on. Returns how many changed."""
        changed = [(new, mid) for mid, old in self.db.execute("SELECT id, content FROM messages")
                   if (new := redact(old)) != old]
        with self.db:
            self.db.executemany("UPDATE messages SET content = ? WHERE id = ?", changed)
        if changed and self._embedder:
            self._embedder.passages([c for c, _ in changed])
        return len(changed)

    def forget_session(self, session_id: str) -> bool:
        self.facts.forget(session_id)
        with self.db:
            return self.db.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount > 0

    def extract_facts(self, *, model: str = FACTS_MODEL, workers: int = 4, extractor: Extractor | None = None,
                      progress=None) -> dict:
        """Run write-time fact extraction (one LLM call per session) for sessions that have none yet.

        Costs API calls, so it only runs when asked. Facts become extra retrieval keys in recall.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        extractor = extractor or Extractor(self.facts, model)
        todo = [x for x in self._sessions() if not self.facts.has(x.session_id)]
        n0, in0, out0 = self.facts.usage()
        failed = []

        def run(sess: RecalledSession):
            rounds = split_rounds([sess.messages])
            facts = extractor.extract(sess.session_id, sess.started_at[:16], [r.user_text for r in rounds])
            if self._embedder and facts:
                self._embedder.passages([f.text for f in facts])

        with ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(run, x): x for x in todo}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:  # keep going; rerun to retry
                    failed.append(f"{futures[fut].session_id}: {type(e).__name__}: {str(e)[:120]}")
                if progress:
                    progress()
        n1, in1, out1 = self.facts.usage()
        return {"sessions": n1 - n0, "failed": failed, "input_tokens": in1 - in0, "output_tokens": out1 - out0,
                "todo": len(todo)}

    # --- reading -------------------------------------------------------------------------

    def _sessions(self) -> list[RecalledSession]:
        sessions = {
            sid: RecalledSession(sid, started, [])
            for sid, started in self.db.execute("SELECT id, started_at FROM sessions ORDER BY started_at, id")
        }
        for row in self.db.execute(
            "SELECT id, session_id, role, content, created_at FROM messages ORDER BY session_id, id"
        ):
            sessions[row[1]].messages.append(Message(*row))
        return [s for s in sessions.values() if s.messages]

    def recall(self, query: str, budget_tokens: int = 8000, *, now: datetime | None = None,
               exclude_sessions: Iterable[str] = ()) -> Recall:
        """The history most relevant to `query` that fits in `budget_tokens`.

        Time words in the query ("上周", "in March", "two weeks ago") are read relative to `now`
        and conversations from that period are ranked higher. `exclude_sessions` leaves out
        sessions already in the assistant's context, such as the current one.
        """
        skip = set(exclude_sessions)
        sessions = [s for s in self._sessions() if s.session_id not in skip]
        rounds = split_rounds([s.messages for s in sessions])
        rid_of = {(r.session_idx, r.round_idx): i for i, r in enumerate(rounds)}
        # Re-pasted text yields the same fact again and again ("started businesses in the Caribbean"
        # from every bio rewrite); identical facts keep one key, the latest, so copies don't crowd recall.
        latest: dict[str, int] = {}
        for si, sess in enumerate(sessions):
            for f in self.facts.get(sess.session_id):
                if (si, f.round_idx) in rid_of:
                    latest[" ".join(f.text.lower().split())] = rid_of[(si, f.round_idx)]
        fact_keys = [(text, rid) for text, rid in latest.items()]
        ref = time_reference(query, now or datetime.now().astimezone())
        window = ref[0] if ref else None
        time_only = bool(ref) and is_time_only(ref[1])
        round_dates = [date.fromisoformat(r.turns[0].created_at[:10]) for r in rounds]
        in_window = sum(window[0] <= d <= window[1] for d in round_dates) if window else 0
        full_rounds = self.full_rounds if self.full_rounds is not None else min(3, budget_tokens // FULL_ROUND_TOKENS)
        picked = select(
            rounds, query, budget_tokens,
            scorer=self.scorer, embedder=self._embedder, full_rounds=full_rounds,
            round_dates=round_dates,
            window=window, window_strict=time_only, reranker=self._reranker,
            # longer turns are excerpted. Once every ranked user turn is in, leftover budget brings
            # replies back under a cap that grows with the budget, so a large budget carries them
            # whole (a fixed 300 had cut the answer out of 9 of the 10 held-out misses at 8k)
            max_turn_tokens=self.max_turn_tokens,
            max_reply_tokens=max(self.max_turn_tokens, budget_tokens // 8) if self.max_turn_tokens else None,
            extra_keys=fact_keys, assistant_keys=self.assistant_keys, reply_lane=self.reply_lane,
            fill_replies=True,
        )
        score = relevance([r.user_text for r in rounds], query, self._embedder) if self._embedder else None

        def build() -> Recall:
            by_session: dict[int, list[Message]] = {}
            for rid in sorted(picked, key=lambda r: (rounds[r].session_idx, rounds[r].round_idx)):
                by_session.setdefault(rounds[rid].session_idx, []).extend(picked[rid])
            first_pick: dict[int, int] = {}
            for order, rid in enumerate(picked):  # dict order = the order select() picked rounds in
                first_pick.setdefault(rounds[rid].session_idx, order)
            rank_of = {si: i + 1 for i, si in enumerate(sorted(first_pick, key=first_pick.get))}
            recalled = [
                RecalledSession(sessions[si].session_id, sessions[si].started_at, msgs, rank_of[si])
                for si, msgs in by_session.items()
            ]
            return Recall(query, recalled, rounds_considered=len(rounds), window=window, time_only=time_only,
                          window_records=in_window, relevance=score)

        result = build()
        # select() estimates formatting per turn; the rendered text (session headers, dated
        # prefixes like "assistant (2026-10-23): ") can come out a few percent longer. Drop the
        # last-picked, least relevant rounds until the real text fits the budget.
        while result.tokens_est > budget_tokens and len(picked) > 1:
            picked.pop(next(reversed(picked)))
            result = build()
        return result

    def for_injection(self, query: str, budget_tokens: int = 1000, *, exclude_sessions: Iterable[str] = (),
                      seen: Iterable[int] = (), now: datetime | None = None) -> dict:
        """What automatic injection would add for `query`: a recall without the current session,
        minus messages already injected earlier in it (`seen`, message ids). If the best-matching
        session brings nothing new, nothing is added: follow-up prompts on the same topic would
        otherwise re-inject the same history every time."""
        r = self.recall(query, budget_tokens, now=now, exclude_sessions=exclude_sessions)
        seen = set(seen)
        fresh = [RecalledSession(x.session_id, x.started_at, [m for m in x.messages if m.id not in seen], x.rank)
                 for x in r.sessions]
        top_is_new = any(x.rank == 1 and x.messages for x in fresh)
        r.sessions = [x for x in fresh if x.messages] if top_is_new else []
        return {"text": r.text if r.sessions else "", "relevance": r.relevance, "time_only": r.time_only,
                "window_records": r.window_records, "ids": [m.id for x in r.sessions for m in x.messages],
                "tokens_est": r.tokens_est if r.sessions else 0}

    def list_sessions(self, *, search: str = "", limit: int = 50, offset: int = 0) -> tuple[int, list[dict]]:
        """(total, page) of sessions, newest first, optionally only those containing `search`."""
        where, args = "", []
        if search:
            where = "WHERE id IN (SELECT session_id FROM messages WHERE content LIKE ?)"
            args.append(f"%{search}%")
        total, = self.db.execute(f"SELECT COUNT(*) FROM sessions {where}", args).fetchone()
        rows = self.db.execute(
            f"""SELECT s.id, s.started_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id),
                       (SELECT content FROM messages m WHERE m.session_id = s.id AND role = 'user' ORDER BY id LIMIT 1)
                FROM sessions s {where} ORDER BY s.started_at DESC, s.id LIMIT ? OFFSET ?""",
            [*args, limit, offset],
        ).fetchall()
        return total, [{"id": i, "started_at": at, "messages": n, "preview": (first or "")[:160]}
                       for i, at, n, first in rows]

    def get_session(self, session_id: str) -> RecalledSession | None:
        row = self.db.execute("SELECT id, started_at FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            return None
        msgs = [Message(*r) for r in self.db.execute(
            "SELECT id, session_id, role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,))]
        return RecalledSession(row[0], row[1], msgs)

    def time_range(self) -> tuple[str, str] | None:
        first, last = self.db.execute("SELECT MIN(created_at), MAX(created_at) FROM messages").fetchone()
        return (first, last) if first else None

    def stats(self) -> dict:
        n_sessions, = self.db.execute("SELECT COUNT(*) FROM sessions").fetchone()
        n_messages, chars = self.db.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(content)), 0) FROM messages").fetchone()
        facts, = self.facts.db.execute("SELECT COUNT(*) FROM facts").fetchone()
        return {"sessions": n_sessions, "messages": n_messages, "tokens_est": int(chars / 2.6),
                "facts": facts, "path": str(self.path)}

    def export_markdown(self) -> str:
        """Everything, readable: memory you can inspect and keep."""
        out = []
        for s in self._sessions():
            out.append(f"## {s.started_at}  ({s.session_id})\n")
            out.extend(f"**{m.role}:** {m.content.strip()}\n" for m in s.messages)
        return "\n".join(out)
