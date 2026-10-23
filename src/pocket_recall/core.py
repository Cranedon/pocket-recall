"""Retrieval core shared by the memory store and the benchmark harness.

User-first selection: rank rounds by what the user said, spend the budget on user turns.
Facts about a person live in their own messages, while assistant replies are long and
generic. So:
  - the top `full_rounds` rounds (BM25 on the whole round) go in whole, so
    "what did you recommend" questions still work;
  - every other round contributes only its user turns, ranked by BM25 and local
    embeddings fused with RRF, which fits many more rounds in the budget.
The budget covers the rendered history, session headers included.

Turns are any objects with `.role` and `.content`; the selection hands back the same
objects, so callers keep whatever else they attach to them.
"""

from __future__ import annotations

import copy
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Protocol, Sequence

import numpy as np

CHARS_PER_TOKEN = 2.6
# Chinese, Japanese and Korean take far more tokens per character than English. Not measured on
# Opus 5.5's tokenizer (the relay used for the benchmarks has no token-count endpoint); 1.5 is a
# deliberately high guess so that CJK packs stay under budget even with an inefficient tokenizer.
CJK_TOKENS_PER_CHAR = 1.5
SESSION_HEADER_TOKENS = 20  # "### Session N:\nSession Date: ...\nSession Content:\n"
TURN_PREFIX_TOKENS = 3  # "\n\nuser: "
RRF_K = 60
ASSISTANT_KEY_TOKENS = 120  # size of the assistant-reply pieces used as retrieval keys
ASSISTANT_EXCERPT_TOKENS = 200  # how much of a reply comes along when one of its pieces matched
RERANK_TOP = 30  # candidates the cross-encoder re-scores; the rest keep their order
RERANK_CHARS = 1500  # cross-encoders read ~512 tokens; long pastes are cut for scoring only
# Character-pair BM25 is noisy for Chinese: pairs like "我的" match nearly every personal
# message. On a 20-question Chinese check, BM25 at full weight pulled top-3 hits from
# 17-19 down to 11-12; at 0.1 it only breaks ties. English keeps equal weights.
CJK_BM25_WEIGHT = 0.1
# Tried 0.5, 1 and 2 on the 62 LongMemEval questions that name a time: evidence fully recalled
# 59, 57, 55 of 62 (56 without the boost). Higher weights bury content matches just outside.
WINDOW_WEIGHT = 0.5


class TurnLike(Protocol):
    role: str
    content: str


class RerankerLike(Protocol):
    def scores(self, query: str, texts: list[str]) -> list[float]: ...


class EmbedderLike(Protocol):
    def passages(self, texts: list[str]) -> np.ndarray: ...
    def query(self, text: str) -> np.ndarray: ...


_CJK_CHAR = re.compile(r"[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]")


def estimate_tokens(text: str) -> int:
    # Used only for packing. Other text: measured on LongMemEval-S history with the Opus 5.5
    # tokenizer at 2.63-3.08 chars/token, so 2.6 keeps packs at or under budget.
    cjk = len(_CJK_CHAR.findall(text))
    return int((len(text) - cjk) / CHARS_PER_TOKEN + cjk * CJK_TOKENS_PER_CHAR) + 1


@dataclass
class Round:
    """One user turn plus the assistant replies that follow it."""

    session_idx: int
    round_idx: int
    turns: list

    @property
    def text(self) -> str:
        return "\n".join(t.content for t in self.turns)

    @property
    def user_text(self) -> str:
        return "\n".join(t.content for t in self.turns if t.role == "user") or self.text

    @property
    def user_turns(self) -> list:
        return [t for t in self.turns if t.role == "user"] or self.turns

    @property
    def assistant_text(self) -> str:
        return "\n".join(t.content for t in self.turns if t.role == "assistant")


def split_rounds(sessions: Sequence[Sequence[TurnLike]]) -> list[Round]:
    rounds: list[Round] = []
    for si, turns in enumerate(sessions):
        groups: list[list] = []
        for turn in turns:
            if turn.role == "user" or not groups:
                groups.append([])
            groups[-1].append(turn)
        rounds.extend(Round(si, ri, g) for ri, g in enumerate(groups))
    return rounds


_WORD = re.compile(r"\w+", re.UNICODE)
_CJK_RUN = re.compile(r"[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]+")


def _cjk_bigrams(text: str) -> str:
    """Chinese/Japanese/Korean have no spaces, so FTS5 would index a whole clause as one word.
    Index overlapping character pairs instead ("我的猫" -> "我的 的猫"); other text is unchanged."""

    def pairs(m: re.Match) -> str:
        run = m.group(0)
        return " " + (" ".join(run[i : i + 2] for i in range(len(run) - 1)) if len(run) > 1 else run) + " "

    return _CJK_RUN.sub(pairs, text)


def has_cjk(text: str) -> bool:
    return _CJK_RUN.search(text) is not None


# Character pairs from Chinese question and function words carry no topic, but they do occur in
# stored messages ("要注意什么"), and a single shared "什么" was enough to outrank the right answer.
_ZH_STOP_PAIRS = set(
    "什么 么时 时候 怎么 么样 么办 为什 哪些 哪个 哪里 哪儿 多少 多久 是不 不是 有没 没有 可以 一下 我的 我们 你的 你们 "
    "一个 这个 那个 这些 那些 了吗 的吗 是什 请问 告诉 知道 记得 之前 以前 的是 了什 做了 过什 在什 都有 有哪 有什".split())


def fts_query(question: str) -> str:
    words = {w.lower() for w in _WORD.findall(_cjk_bigrams(question)) if len(w) > 1} - _ZH_STOP_PAIRS
    return " OR ".join(f'"{w}"' for w in sorted(words)) or '""'


def bm25_rank(texts: list[str], query: str) -> list[int]:
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE u USING fts5(body, tokenize='porter unicode61')")
    db.executemany("INSERT INTO u(rowid, body) VALUES (?, ?)", [(i, _cjk_bigrams(t)) for i, t in enumerate(texts)])
    ranked = [r[0] for r in db.execute("SELECT rowid FROM u WHERE u MATCH ? ORDER BY bm25(u)", (fts_query(query),))]
    db.close()
    return ranked


def dense_rank(texts: list[str], query: str, embedder: EmbedderLike) -> list[int]:
    scores = embedder.passages(texts) @ embedder.query(query)
    return [int(i) for i in np.argsort(-scores)]


def rrf(*rankings: list[int], weights: Sequence[float] | None = None) -> list[int]:
    score: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights or [1.0] * len(rankings)):
        for pos, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + weight / (RRF_K + pos + 1)
    return sorted(score, key=lambda i: -score[i])


def rank(texts: list[str], query: str, scorer: str = "hybrid", embedder: EmbedderLike | None = None,
         reranker: RerankerLike | None = None) -> list[int]:
    """Every index exactly once, best first. BM25 skips texts with no word overlap; those trail.
    With a `reranker`, the top RERANK_TOP are re-ordered by its scores."""
    if not texts:
        return []
    if scorer == "bm25":
        ranked = bm25_rank(texts, query)
    elif scorer == "dense":
        ranked = dense_rank(texts, query, embedder)
    else:
        bm25_weight = CJK_BM25_WEIGHT if has_cjk(query) else 1.0
        ranked = rrf(bm25_rank(texts, query), dense_rank(texts, query, embedder), weights=[bm25_weight, 1.0])
    seen = set(ranked)
    ranked += [i for i in range(len(texts)) if i not in seen]
    if reranker is not None and len(ranked) > 1:
        top = ranked[:RERANK_TOP]
        scores = reranker.scores(query, [texts[i][:RERANK_CHARS] for i in top])
        ranked = [i for _, i in sorted(zip(scores, top), key=lambda x: -x[0])] + ranked[RERANK_TOP:]
    return ranked


_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])|(?<=[.!?])\s+|\n+")
EXCERPT_CHUNK_TOKENS = 60


def chunk_text(text: str, chunk_tokens: int) -> list[str]:
    """Consecutive sentences grouped into pieces of about `chunk_tokens`."""
    chunks: list[str] = []
    for sentence in (s for s in _SENTENCE_END.split(text) if s and s.strip()):
        if chunks and estimate_tokens(chunks[-1]) + estimate_tokens(sentence) <= chunk_tokens:
            chunks[-1] += " " + sentence.strip()
        else:
            sentence = sentence.strip()
            # an overlong "sentence" (a pasted log, a run-on line) is cut to about two chunks
            limit = int(2 * chunk_tokens * len(sentence) / estimate_tokens(sentence))
            chunks.append(sentence[: max(limit, 1)])
    return chunks


def excerpt(text: str, query: str, max_tokens: int, *, scorer: str = "hybrid",
            embedder: EmbedderLike | None = None) -> str:
    """The parts of a long message most relevant to `query`, in original order, within `max_tokens`.

    Real users paste whole résumés and emails; one message can exceed the whole budget.
    The text is cut into chunks of a few sentences; the opening chunk is always kept for context.
    """
    if estimate_tokens(text) <= max_tokens:
        return text
    chunks = chunk_text(text, EXCERPT_CHUNK_TOKENS)
    keep, used = {0}, estimate_tokens(chunks[0])
    for i in rank(chunks, query, scorer, embedder):
        cost = estimate_tokens(chunks[i]) + 1
        if i not in keep and used + cost <= max_tokens:
            keep.add(i)
            used += cost
    parts, prev = [], -1
    for i in sorted(keep):
        if prev >= 0 and i != prev + 1:
            parts.append("…")
        parts.append(chunks[i])
        prev = i
    if prev != len(chunks) - 1:
        parts.append("…")
    return " ".join(parts)


def select(
    rounds: list[Round],
    query: str,
    budget: int,
    *,
    scorer: str = "hybrid",
    embedder: EmbedderLike | None = None,
    full_rounds: int = 3,
    extra_keys: Sequence[tuple[str, int]] = (),
    round_dates: Sequence[date] | None = None,
    window: tuple[date, date] | None = None,
    window_weight: float = WINDOW_WEIGHT,
    window_strict: bool = False,
    max_turn_tokens: int | None = None,
    reranker: RerankerLike | None = None,
    assistant_keys: bool = False,
) -> dict[int, list]:
    """Pick rounds for `query` within `budget` tokens: round index -> turns to include.

    `extra_keys` are additional (text, round index) retrieval keys, e.g. extracted facts:
    a round can be found through any of its keys, but what is included is its own turns.
    `window` (from `timeparse`) with `round_dates` ranks rounds dated inside the window higher:
    a soft boost by default, or strictly first with `window_strict`, for questions that name
    only a time ("what did I do last week") where the content ranking is noise.
    `max_turn_tokens` replaces longer turns with their most relevant excerpt (copies; the
    originals are untouched), so one pasted document cannot crowd out everything else.
    `assistant_keys` also ranks pieces of the assistant's replies: a round found that way comes
    with an excerpt of the reply. In chat logs the answer is often the assistant's conclusion,
    which user-first packing would otherwise only see through the few whole rounds.
    """

    def shorten(turns: list) -> list:
        if not max_turn_tokens:
            return turns
        out = []
        for t in turns:
            if estimate_tokens(t.content) > max_turn_tokens:
                t = copy.copy(t)
                t.content = excerpt(t.content, query, max_turn_tokens, scorer=scorer, embedder=embedder)
            out.append(t)
        return out


    def outside(rid: int) -> bool:  # sort key: rounds inside the window first, order otherwise kept
        return window is not None and round_dates is not None and not window[0] <= round_dates[rid] <= window[1]

    used = 0
    sessions_touched: set[int] = set()
    picked: dict[int, list] = {}

    def take(rid: int, turns: list) -> bool:
        """Add `turns` for round `rid` (to what is already picked for it) if they fit."""
        nonlocal used
        header = 0 if rounds[rid].session_idx in sessions_touched else SESSION_HEADER_TOKENS
        cost = header + sum(estimate_tokens(t.content) + TURN_PREFIX_TOKENS for t in turns)
        if used + cost > budget:
            return False
        picked[rid] = picked.get(rid, []) + turns
        used += cost
        sessions_touched.add(rounds[rid].session_idx)
        return True

    def reply_excerpt(rid: int) -> list:
        """The round's assistant reply cut to the part about the question, as one turn."""
        replies = [t for t in rounds[rid].turns if t.role == "assistant"]
        if not replies:
            return []
        full = rounds[rid].assistant_text
        t = copy.copy(replies[0])
        t.content = excerpt(full, query, ASSISTANT_EXCERPT_TOKENS, scorer=scorer, embedder=embedder)
        if t.content != full and hasattr(t, "has_answer"):
            t.has_answer = False  # benchmark bookkeeping: a cut reply does not count as recalled evidence
        return [t]

    texts = [r.user_text for r in rounds] + [text for text, _ in extra_keys]
    owners = list(range(len(rounds))) + [rid for _, rid in extra_keys]
    from_reply = [False] * len(texts)
    if assistant_keys:
        for rid, r in enumerate(rounds):
            for piece in chunk_text(r.assistant_text, ASSISTANT_KEY_TOKENS) if r.assistant_text else []:
                texts.append(piece)
                owners.append(rid)
                from_reply.append(True)
    ranked = rank(texts, query, scorer, embedder, reranker)
    if window_strict:
        ranked = sorted(ranked, key=lambda k: outside(owners[k]))
    elif window is not None and round_dates is not None:
        # Soft boost: rounds dated in the window get an extra RRF vote, so they rise
        # without burying strong content matches from just outside it.
        ranked = rrf(ranked, [k for k in ranked if not outside(owners[k])], weights=[1.0, window_weight])

    # Whole rounds, for answers that live in an assistant reply. English: BM25 on the full
    # round text. Chinese: that BM25 is too noisy, so take the best rounds of the main ranking.
    if has_cjk(query) and scorer != "bm25":
        candidates = list(dict.fromkeys(owners[k] for k in ranked))
    else:
        candidates = bm25_rank([r.text for r in rounds], query)
    if window_strict:
        candidates = sorted(candidates, key=outside)
    elif window is not None and round_dates is not None:
        candidates = rrf(candidates, [r for r in candidates if not outside(r)], weights=[1.0, window_weight])
    whole = 0
    for rid in candidates[: full_rounds * 2] if full_rounds else []:  # only the best; no filling with weak rounds
        if whole >= full_rounds:
            break
        whole += take(rid, shorten(rounds[rid].turns))

    # Everything else: user turns, ranked on what the user said (or keys drawn from it); a round
    # matched through a piece of its reply also gets that reply's relevant excerpt.
    for key in ranked:
        rid = owners[key]
        if from_reply[key]:
            has_reply = any(t.role == "assistant" for t in picked.get(rid, []))
            if rid not in picked:
                take(rid, shorten(rounds[rid].user_turns) + reply_excerpt(rid))
            elif not has_reply:
                take(rid, reply_excerpt(rid))
        elif rid not in picked:
            take(rid, shorten(rounds[rid].user_turns))
    return picked
