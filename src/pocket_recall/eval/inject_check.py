"""Measure automatic injection: does the gate fire when memory has the answer, stay quiet when it
does not, and does the injected budget carry the evidence?

For every question, "with memory" asks it against its own history; "without memory" asks it
against the history of a different question (LongMemEval) or a different conversation (LoCoMo),
which has the same size and style but not the answer. Recall uses the injection budget and no
whole rounds, as the hook does.

    pocket-recall eval inject [--datasets lme locomo-dev] [--budget 1000] [--threshold 0.6]
    pocket-recall eval inject --replay ~/.pocket-recall/memory.db

`--replay` plays a store's own sessions back in order, each prompt seeing every other session as
memory, through the hook's full logic (gate, time-only pass, no repeats within a session). It
shows how often injection would fire in real use and what it costs in tokens.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from ..core import relevance
from ..inject import DEFAULT_BUDGET, DEFAULT_THRESHOLD
from ..store import DEFAULT_EMBED_MODEL
from . import data, locomo
from .baselines import _split_rounds, make_method

LOADERS = {
    "lme": lambda: data.load("s"),
    "locomo-dev": lambda: locomo.load("dev")[0],
    "locomo-test": lambda: locomo.load("test")[0],
}


def _conv(inst) -> str:
    return inst.question_id.split("-q")[0]


def check(name: str, budget: int, threshold: float, n: int | None, seed: int = 0) -> str:
    from ..embed import Embedder
    from . import userfirst

    userfirst._embedder = Embedder(data.DATA_DIR / "embeddings.sqlite", DEFAULT_EMBED_MODEL)
    emb = userfirst._embedder
    pool = [i for i in LOADERS[name]() if not i.is_abstention]
    rng = random.Random(seed)
    inst = rng.sample(pool, n) if n and n < len(pool) else pool
    method = make_method("userfirst-time-hybrid-f0", budget)
    fired_pos = fired_neg = carried = 0
    for k, i in enumerate(tqdm(inst, desc=name, leave=False)):
        if name == "lme":
            other = inst[(k + 7) % len(inst)]
        else:
            others = [x for x in inst if _conv(x) != _conv(i)]
            other = others[k % len(others)]
        own = relevance([r.user_text for r in _split_rounds(i.sessions)], i.question, emb)
        alien = relevance([r.user_text for r in _split_rounds(other.sessions)], i.question, emb)
        fired_neg += alien >= threshold
        if own >= threshold:
            fired_pos += 1
            s = method.build(i).stats
            carried += s["evidence_turns_hit"] == s["evidence_turns_total"]
    m = len(inst)
    return (f"| {name} | {m} | {fired_pos / m:.1%} | {fired_neg / m:.1%} | "
            f"{carried / max(fired_pos, 1):.1%} | {carried / m:.1%} |")


def replay(path, budget: int, threshold: float) -> str:
    from datetime import datetime

    from ..inject import MIN_PROMPT_CHARS, passes
    from ..store import MemoryStore

    store = MemoryStore(path)
    prompts = fired = tokens = 0
    for sess in store._sessions():
        seen: list[int] = []
        for m in sess.messages:
            if m.role != "user" or len(m.content.strip()) < MIN_PROMPT_CHARS:
                continue
            prompts += 1
            now = datetime.fromisoformat(m.created_at)
            r = store.for_injection(m.content[:2000], budget, exclude_sessions=[sess.session_id], seen=seen, now=now)
            if passes(r, threshold):
                fired += 1
                tokens += r["tokens_est"]
                seen += r["ids"]
    store.close()
    return (f"| {Path(path).name} | {prompts} | {fired / max(prompts, 1):.0%} | "
            f"{tokens / max(prompts, 1):,.0f} | {tokens / max(fired, 1):,.0f} |")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pocket-recall eval inject", description=__doc__.split("\n\n")[0])
    p.add_argument("--datasets", nargs="+", default=["lme", "locomo-dev"], choices=list(LOADERS))
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--n", type=int, help="sample this many questions per dataset (default: all)")
    p.add_argument("--replay", nargs="+", type=Path, help="stores to replay instead of the benchmarks")
    args = p.parse_args(argv)
    if args.replay:
        print(f"Replay at {args.budget} tokens, gate {args.threshold}\n")
        print("| store | prompts | prompts that inject | tokens per prompt | tokens per injection |")
        print("|---|---:|---:|---:|---:|")
        for path in args.replay:
            print(replay(path.expanduser(), args.budget, args.threshold), flush=True)
        return
    print(f"Injection at {args.budget} tokens, gate threshold {args.threshold}\n")
    print("| dataset | questions | fires with memory | fires without memory | evidence complete when fired | evidence injected overall |")
    print("|---|---:|---:|---:|---:|---:|")
    for name in args.datasets:
        print(check(name, args.budget, args.threshold, args.n), flush=True)


if __name__ == "__main__":
    main()
