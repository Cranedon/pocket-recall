"""Real chat histories for hands-on testing, from WildChat-1M (allenai, ODC-BY).

WildChat logs real people's conversations with ChatGPT, with consent and PII redacted. Each
conversation has a timestamp and an anonymised user id (hashed IP), so one person's history
over weeks or months can be rebuilt: the user side is real, the assistant side is ChatGPT.

    uv run --with duckdb python -m pocket_recall.eval.wildchat candidates --lang Chinese
    uv run --with duckdb python -m pocket_recall.eval.wildchat show <hashed_ip>
    uv run --with duckdb python -m pocket_recall.eval.wildchat import <hashed_ip> ~/.pocket-recall/demo-x.db

Needs data/wildchat/meta.duckdb with a `convs` table (see README: 真实对话数据).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .data import DATA_DIR

DB = DATA_DIR / "wildchat" / "meta.duckdb"
_CODE = re.compile(r"```|\bdef \w+\(|\bfunction\b|#include|\bimport \w+|SELECT .* FROM|</?\w+>|\{\s*\"")
_PERSONAL = re.compile(r"\b(my|I'm|I am|I've|I was|me)\b|我的|我是|我在|我想|我们|我家|我今天|我最近", re.I)


def connect():
    import duckdb

    return duckdb.connect(str(DB), read_only=True)


def user_turns(conversation) -> list[str]:
    return [t["content"] for t in conversation if t["role"] == "user" and t["content"]]


def candidates(lang: str, top: int) -> None:
    """Rank users by how personal and non-technical their messages are."""
    con = connect()
    rows = con.execute(
        "SELECT c.hashed_ip, u.n, u.active_days, u.span_days, c.conversation FROM convs c "
        "JOIN cand u USING (hashed_ip) WHERE u.lang = ?", [lang]).fetchall()
    per_user: dict[str, dict] = {}
    for ip, n, days, span, conv in rows:
        u = per_user.setdefault(ip, {"n": n, "days": days, "span": span, "msgs": 0, "personal": 0, "code": 0,
                                     "distinct": set()})
        for text in user_turns(conv):
            u["msgs"] += 1
            u["personal"] += bool(_PERSONAL.search(text))
            u["code"] += bool(_CODE.search(text))
            u["distinct"].add(text[:40])
    scored = []
    for ip, u in per_user.items():
        if u["msgs"] < 40:
            continue
        variety = len(u["distinct"]) / u["msgs"]  # scripted users repeat the same prompt
        score = (u["personal"] / u["msgs"]) * (1 - u["code"] / u["msgs"]) * variety
        scored.append((score, ip, u))
    for score, ip, u in sorted(scored, reverse=True)[:top]:
        print(f"{score:.2f}  {ip}  convs={u['n']} active_days={u['days']} span={u['span']}d "
              f"personal={u['personal']}/{u['msgs']} code={u['code']} variety={len(u['distinct'])}/{u['msgs']}")


def show(ip: str, limit: int) -> None:
    con = connect()
    for ts, conv in con.execute(
            "SELECT timestamp, conversation FROM convs WHERE hashed_ip = ? ORDER BY timestamp LIMIT ?",
            [ip, limit]).fetchall():
        turns = user_turns(conv)
        print(f"{ts:%Y-%m-%d %H:%M}  [{len(turns)} msgs]  {turns[0][:150].replace(chr(10), ' ') if turns else ''}")


def import_user(ip: str, out: Path) -> None:
    from ..store import MemoryStore

    con = connect()
    rows = con.execute(
        "SELECT conversation_hash, timestamp, conversation FROM convs WHERE hashed_ip = ? ORDER BY timestamp",
        [ip]).fetchall()
    store = MemoryStore(out)
    existing = {s["id"] for s in store.list_sessions(limit=10**6)[1]}
    added = 0
    for conv_hash, ts, conv in rows:
        sid = conv_hash[:12]
        if sid in existing:
            continue
        messages = [{"role": t["role"], "content": t["content"],
                     "created_at": (t["timestamp"] or ts).isoformat()} for t in conv if t["content"]]
        store.add_session(messages, session_id=sid, started_at=ts.isoformat())
        added += 1
    print(f"imported {added} conversations into {out} ({store.stats()['messages']} messages)")
    store.close()


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m pocket_recall.eval.wildchat")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("candidates")
    sp.add_argument("--lang", default="Chinese")
    sp.add_argument("--top", type=int, default=15)
    sp = sub.add_parser("show")
    sp.add_argument("ip")
    sp.add_argument("--limit", type=int, default=400)
    sp = sub.add_parser("import")
    sp.add_argument("ip")
    sp.add_argument("out", type=Path)
    args = p.parse_args()
    if args.cmd == "candidates":
        candidates(args.lang, args.top)
    elif args.cmd == "show":
        show(args.ip, args.limit)
    else:
        import_user(args.ip, args.out.expanduser())


if __name__ == "__main__":
    main()
