"""Run the example questions of demo stores and check the right messages come back.

Each <store>.questions.json next to a store lists questions with the words the answer
must contain ("expect"), written from the stored conversations themselves. A question
passes when every expected word appears in what recall returns.

    uv run python -m pocket_recall.eval.demo_check [--dir ~/.pocket-recall/demo] [--budgets 2000 8000]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ..store import MemoryStore


def check(directory: Path, budgets: list[int], verbose: bool, only: list[str] | None = None) -> None:
    totals = {b: [0, 0] for b in budgets}
    for qfile in sorted(directory.glob("*.questions.json")):
        if only and qfile.name.replace(".questions.json", "") not in only:
            continue
        store = MemoryStore(directory / qfile.name.replace(".questions.json", ".db"))
        questions = json.loads(qfile.read_text())
        print(f"\n== {qfile.name.replace('.questions.json', '')}  ({store.stats()['tokens_est']:,} tokens stored)")
        print("   " + "".join(f"{b:>8}" for b in budgets) + "   question")
        for item in questions:
            now = datetime.fromisoformat(item["now"]).astimezone()
            cells, misses = [], {}
            for b in budgets:
                r = store.recall(item["q"], b, now=now)
                text = r.text.lower()
                found = [w for w in item["expect"] if w.lower() in text]
                totals[b][0] += len(found) == len(item["expect"])
                totals[b][1] += 1
                cells.append("ok" if len(found) == len(item["expect"]) else f"{len(found)}/{len(item['expect'])}")
                if len(found) < len(item["expect"]):
                    misses[b] = [w for w in item["expect"] if w not in found]
            print("   " + "".join(f"{c:>8}" for c in cells) + f"   {item['q']}")
            if verbose and misses:
                print("   " + " " * 8 * len(budgets) + f"   missing: {misses}  (answer: {item['answer']}; in: {item['where']})")
        store.close()
    print("\nall expected words recalled: " + ", ".join(f"{b} tokens {ok}/{n}" for b, (ok, n) in totals.items()))


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m pocket_recall.eval.demo_check")
    p.add_argument("--dir", default="~/.pocket-recall/demo")
    p.add_argument("--budgets", type=int, nargs="+", default=[2000, 8000])
    p.add_argument("-v", "--verbose", action="store_true", help="show what is missing")
    p.add_argument("--stores", nargs="+", help="only these stores (names without .db)")
    args = p.parse_args()
    check(Path(args.dir).expanduser(), args.budgets, args.verbose, args.stores)


if __name__ == "__main__":
    main()
