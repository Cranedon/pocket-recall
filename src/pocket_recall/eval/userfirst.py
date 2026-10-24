"""Benchmark wrapper around the shared user-first retrieval in `pocket_recall.core`."""

from __future__ import annotations

from datetime import date, datetime

from tqdm import tqdm

from ..core import ASSISTANT_KEY_TOKENS, REPLY_LANE_SHARE, chunk_text, select
from ..embed import MODEL, Embedder
from ..facts import FactStore
from ..timeparse import is_time_only, time_reference
from .baselines import Context, _evidence_stats, _split_rounds
from .data import DATA_DIR, Instance, Session

_embedder: Embedder | None = None
_facts: FactStore | None = None


def embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder(DATA_DIR / "embeddings.sqlite", MODEL)
    return _embedder


def fact_store() -> FactStore:
    global _facts
    if _facts is None:
        _facts = FactStore(DATA_DIR / "facts.sqlite")
    return _facts


def user_text(rnd) -> str:
    return rnd.user_text


def parse_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y/%m/%d").date()  # "2023/05/20 (Sat) 02:21"


class UserFirstRetrieval:
    def __init__(self, budget_tokens: int = 8000, scorer: str = "hybrid", full_rounds: int = 3, facts: bool = False,
                 time: bool = False, assistant: bool = False, lane: bool = False, fill: bool = False):
        self.budget = budget_tokens
        self.scorer = scorer
        self.full_rounds = full_rounds
        self.facts = facts
        self.time = time
        self.assistant = assistant
        self.lane = lane
        self.fill = fill
        flags = (("facts-" if facts else "") + ("time-" if time else "") + ("asst-" if assistant else "")
                 + ("lane-" if lane else "") + ("fill-" if fill else ""))
        self.name = f"userfirst-{flags}{scorer}-f{full_rounds}-{budget_tokens}"

    def fact_keys(self, inst: Instance, rounds) -> list[tuple[str, int]]:
        """Extracted facts as extra retrieval keys, each pointing at the round it came from."""
        if not self.facts:
            return []
        rid_of = {(r.session_idx, r.round_idx): i for i, r in enumerate(rounds)}
        return [
            (fact.text, rid_of[(si, fact.round_idx)])
            for si, sess in enumerate(inst.sessions)
            for fact in fact_store().get(sess.session_id)
            if (si, fact.round_idx) in rid_of
        ]

    def covered(self, inst: Instance) -> bool:
        """With facts on, a question is only fair to score once every session in it was extracted."""
        return not self.facts or all(fact_store().has(s.session_id) for s in inst.sessions)

    def warm(self, instances: list[Instance]) -> None:
        """Embed every retrieval key up front in big batches; later lookups hit the cache."""
        if self.scorer == "bm25":
            return
        texts: set[str] = set()
        for inst in instances:
            rounds = _split_rounds(inst.sessions)
            texts.update(r.user_text for r in rounds)
            texts.update(text for text, _ in self.fact_keys(inst, rounds))
            if self.assistant or self.lane:
                texts.update(p for r in rounds if r.assistant_text for p in chunk_text(r.assistant_text, ASSISTANT_KEY_TOKENS))
        ordered = sorted(texts, key=len)
        # Sorted by length so each batch pads to similar lengths; committed in chunks so a stop loses little.
        for i in tqdm(range(0, len(ordered), 4096), desc="embedding", leave=False):
            embedder().passages(ordered[i : i + 4096])

    def build(self, inst: Instance) -> Context:
        rounds = _split_rounds(inst.sessions)
        ref = time_reference(inst.question, parse_date(inst.question_date)) if self.time else None
        picked = select(
            rounds,
            inst.question,
            self.budget,
            scorer=self.scorer,
            embedder=embedder() if self.scorer != "bm25" else None,
            full_rounds=self.full_rounds,
            extra_keys=self.fact_keys(inst, rounds),
            round_dates=[parse_date(inst.sessions[r.session_idx].date) for r in rounds] if self.time else None,
            window=ref[0] if ref else None,
            window_strict=bool(ref) and is_time_only(ref[1]),
            assistant_keys=self.assistant,
            reply_lane=REPLY_LANE_SHARE if self.lane else 0.0,
            fill_replies=self.fill,
        )
        by_session: dict[int, list] = {}
        for rid in sorted(picked, key=lambda r: (rounds[r].session_idx, rounds[r].round_idx)):
            by_session.setdefault(rounds[rid].session_idx, []).extend(picked[rid])
        sessions = [
            Session(inst.sessions[si].session_id, inst.sessions[si].date, turns) for si, turns in by_session.items()
        ]
        sessions.sort(key=lambda s: s.date)
        stats = _evidence_stats(inst, sessions)
        stats["rounds_retrieved"] = len(picked)
        stats["rounds_total"] = len(rounds)
        return Context(sessions, stats)
