"""Baseline memory methods. Each one decides which history goes into the prompt.

- full:  every session, in date order (the ceiling a long-context model reaches by brute force)
- bm25:  raw rounds retrieved with SQLite FTS5 BM25, packed into a fixed token budget
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core import Round, bm25_rank, estimate_tokens, split_rounds
from .data import Instance, Session
from .prompts import format_history


@dataclass
class Context:
    sessions: list[Session]
    stats: dict = field(default_factory=dict)


def _evidence_stats(inst: Instance, sessions: list[Session]) -> dict:
    kept_ids = {s.session_id for s in sessions}
    answer_ids = set(inst.answer_session_ids)
    evidence_total = sum(t.has_answer for s in inst.sessions for t in s.turns)
    evidence_kept = sum(t.has_answer for s in sessions for t in s.turns)
    return {
        "answer_sessions_hit": len(answer_ids & kept_ids),
        "answer_sessions_total": len(answer_ids),
        "evidence_turns_hit": evidence_kept,
        "evidence_turns_total": evidence_total,
        "history_tokens_est": estimate_tokens(format_history(sessions)),  # rendered, headers included
    }


class FullContext:
    name = "full"

    def build(self, inst: Instance) -> Context:
        sessions = sorted(inst.sessions, key=lambda s: s.date)
        return Context(sessions, _evidence_stats(inst, sessions))


def _split_rounds(sessions: list[Session]) -> list[Round]:
    return split_rounds([s.turns for s in sessions])


class BM25Retrieval:
    def __init__(self, budget_tokens: int = 8000):
        self.budget = budget_tokens
        self.name = f"bm25-{budget_tokens}"

    def build(self, inst: Instance) -> Context:
        rounds = _split_rounds(inst.sessions)
        ranked = bm25_rank([r.text for r in rounds], inst.question)

        chosen: list[int] = []
        used = 0
        for rid in ranked:
            cost = estimate_tokens(rounds[rid].text)
            if used + cost > self.budget:
                continue  # a smaller round further down may still fit
            chosen.append(rid)
            used += cost

        # Regroup the chosen rounds into their sessions, keeping original order, then sort sessions by date.
        by_session: dict[int, list[Round]] = {}
        for rid in sorted(chosen, key=lambda r: (rounds[r].session_idx, rounds[r].round_idx)):
            by_session.setdefault(rounds[rid].session_idx, []).append(rounds[rid])
        sessions = [
            Session(
                session_id=inst.sessions[si].session_id,
                date=inst.sessions[si].date,
                turns=[t for r in rs for t in r.turns],
            )
            for si, rs in by_session.items()
        ]
        sessions.sort(key=lambda s: s.date)
        stats = _evidence_stats(inst, sessions)
        stats["rounds_retrieved"] = len(chosen)
        stats["rounds_total"] = len(rounds)
        return Context(sessions, stats)


def make_method(name: str, budget: int):
    if name == "full":
        return FullContext()
    if name == "bm25":
        return BM25Retrieval(budget)
    if name.startswith("userfirst"):
        # userfirst[-facts][-time][-asst][-lane][-fill][-<scorer>[-f<full rounds>]], e.g. userfirst-time-hybrid-f3
        from .userfirst import UserFirstRetrieval

        parts = name.split("-")[1:]
        flags = set()
        while parts and parts[0] in ("facts", "time", "asst", "lane", "fill"):
            flags.add(parts.pop(0))
        scorer = parts[0] if parts else "hybrid"
        full = int(parts[1][1:]) if len(parts) > 1 else 3
        return UserFirstRetrieval(budget, scorer, full, "facts" in flags, "time" in flags, "asst" in flags,
                                  "lane" in flags, "fill" in flags)
    raise ValueError(f"unknown method {name!r}")
