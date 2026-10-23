"""Run a memory method on LongMemEval and score it with the official judge prompts.

Results go to results/<run>.jsonl, one line per question, so runs resume after interruption.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm

from . import data
from .baselines import estimate_tokens, make_method
from .llm import LLM, ClaudeLLM, EchoLLM
from .prompts import answer_prompt, judge_prompt, parse_judgement
from ..store import DEFAULT_EMBED_MODEL as STORE_EMBED_MODEL

RESULTS_DIR = data.DATA_DIR.parent / "results"

# $ per million tokens (input, output). Haiku 5.5 doubles past 100K-token prompts; not modelled here.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}

ANSWER_MAX_TOKENS = 16000
JUDGE_MAX_TOKENS = 2000


def make_llm(model: str, effort: str, fallbacks: bool) -> LLM:
    return EchoLLM() if model == "echo" else ClaudeLLM(model, effort=effort, fallbacks=fallbacks)


def cost(model: str, input_tokens: int, cache_write: int, cache_read: int, output_tokens: int) -> float:
    """List-price equivalent; cache writes bill at 1.25x input, cache reads at 0.1x."""
    pin, pout = PRICES.get(model, (0.0, 0.0))
    uncached = input_tokens - cache_write - cache_read
    return ((uncached + 1.25 * cache_write + 0.1 * cache_read) * pin + output_tokens * pout) / 1e6


def evaluate_one(inst: data.Instance, method, answerer: LLM, judge: LLM, style: str = "official") -> dict:
    ctx = method.build(inst)
    prompt = answer_prompt(ctx.sessions, inst.question_date, inst.question, style)
    ans = answerer.complete(prompt, max_tokens=ANSWER_MAX_TOKENS)
    jprompt = judge_prompt(inst.question_type, inst.question, inst.answer, ans.text, inst.is_abstention)
    verdict = judge.complete(jprompt, max_tokens=JUDGE_MAX_TOKENS)
    return {
        "question_id": inst.question_id,
        "category": inst.category,
        "question": inst.question,
        "gold": inst.answer,
        "hypothesis": ans.text,
        "correct": parse_judgement(verdict.text),
        "judge_reply": verdict.text,
        "endpoint": urlparse(os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")).netloc,
        "answer_model_requested": answerer.name,
        "answer_model": ans.model,
        "answer_stop_reason": ans.stop_reason,
        "answer_prompt_tokens_est": estimate_tokens(prompt),
        "answer_input_tokens": ans.input_tokens,
        "answer_output_tokens": ans.output_tokens,
        "answer_cache_write_tokens": ans.cache_write_tokens,
        "answer_cache_read_tokens": ans.cache_read_tokens,
        "judge_model_requested": judge.name,
        "judge_model": verdict.model,
        "judge_prompt_tokens_est": estimate_tokens(jprompt),
        "judge_input_tokens": verdict.input_tokens,
        "judge_output_tokens": verdict.output_tokens,
        "judge_cache_write_tokens": verdict.cache_write_tokens,
        "judge_cache_read_tokens": verdict.cache_read_tokens,
        **ctx.stats,
    }


def select(args, instances: list[data.Instance]) -> list[data.Instance]:
    """Apply --categories and --qids filters."""
    if getattr(args, "categories", None):
        instances = [i for i in instances if i.category in args.categories.split(",")]
    if getattr(args, "qids", None):
        wanted = set(Path(args.qids).read_text().split())
        instances = [i for i in instances if i.question_id in wanted]
    return instances


def drop_uncovered(method, instances: list[data.Instance]) -> list[data.Instance]:
    if not hasattr(method, "covered"):
        return instances
    kept = [i for i in instances if method.covered(i)]
    if len(kept) < len(instances):
        print(f"skipping {len(instances) - len(kept)} questions whose sessions have no extracted facts yet")
    return kept


def _inflated(row: dict, role: str) -> bool:
    """API-reported prompt size far above what we sent: the relay misreported or added content."""
    if role == "answer":
        sent = row.get("answer_prompt_tokens_est", row["history_tokens_est"])
    else:
        sent = row.get("judge_prompt_tokens_est", 1000)
    return row[f"{role}_input_tokens"] > 1.5 * sent + 1000


def _load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {r["question_id"]: r for r in map(json.loads, f) if "error" not in r}


def cmd_run(args) -> None:
    instances = select(args, data.stratified_sample(data.load(args.split), args.n, args.seed))
    method = make_method(args.method, args.budget)
    instances = drop_uncovered(method, instances)
    if hasattr(method, "warm"):
        method.warm(instances)
    variant = "" if args.answer_style == "official" else f"-{args.answer_style}"
    variant += "" if args.effort == "medium" else f"-e{args.effort}"
    run_name = args.name or f"{args.split}-{method.name}{variant}-{args.answer_model}-n{len(instances)}"
    out = RESULTS_DIR / f"{run_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = _load_done(out)
    todo = [i for i in instances if i.question_id not in done]
    print(f"run {run_name}: {len(instances)} questions, {len(done)} already done, {len(todo)} to go")

    answerer = make_llm(args.answer_model, args.effort, not args.no_fallbacks)
    judge = make_llm(args.judge_model, args.judge_effort, not args.no_fallbacks)
    lock = threading.Lock()
    with out.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(evaluate_one, inst, method, answerer, judge, args.answer_style): inst for inst in todo}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            inst = futures[fut]
            try:
                row = fut.result()
            except Exception as e:  # keep going; failed rows are retried on the next run
                row = {"question_id": inst.question_id, "error": f"{type(e).__name__}: {e}"}
                tqdm.write(f"{inst.question_id}: {row['error']}")
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    report(out, args.answer_model, args.judge_model)


def report(path: Path, answer_model: str | None = None, judge_model: str | None = None) -> None:
    with path.open() as f:
        rows = list(map(json.loads, f))
    errors = {r["question_id"] for r in rows if "error" in r}
    ok = {r["question_id"]: r for r in rows if "error" not in r}
    errors -= ok.keys()
    rows = list(ok.values())
    if not rows:
        print("no scored rows yet")
        return

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    print(f"\n{path.name}")
    print(f"{'category':<28}{'n':>4}{'acc':>8}{'ev.recall':>11}")
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        acc = sum(r["correct"] for r in rs) / len(rs)
        total = sum(r["evidence_turns_total"] for r in rs)
        rec = sum(r["evidence_turns_hit"] for r in rs) / total if total else float("nan")
        print(f"{cat:<28}{len(rs):>4}{acc:>8.1%}{rec:>11.1%}")
    overall = sum(r["correct"] for r in rows) / len(rows)
    task_avg = sum(sum(r["correct"] for r in rs) / len(rs) for rs in by_cat.values()) / len(by_cat)
    print(f"{'overall (question-avg)':<28}{len(rows):>4}{overall:>8.1%}")
    print(f"{'overall (task-avg)':<28}{'':>4}{task_avg:>8.1%}")

    def total(role: str, field: str) -> int:
        return sum(r.get(f"{role}_{field}", 0) for r in rows)

    am = answer_model or rows[0]["answer_model_requested"]
    jm = judge_model or rows[0]["judge_model_requested"]
    ain = total("answer", "input_tokens")
    median = sorted(r["answer_input_tokens"] for r in rows)[len(rows) // 2]
    print(f"\nanswer prompt tokens / question: mean {ain / len(rows):,.0f}, median {median:,}  "
          "(API-reported, cache included)")
    bad = {role: [r["question_id"] for r in rows if _inflated(r, role)] for role in ("answer", "judge")}
    if bad["answer"] or bad["judge"]:
        print(f"usage anomalies: {len(bad['answer'])} answer / {len(bad['judge'])} judge requests were reported "
              "as far larger than the prompt sent (seen with relays); costs below take those reports at face value")
    acost = cost(am, ain, total("answer", "cache_write_tokens"), total("answer", "cache_read_tokens"),
                 total("answer", "output_tokens"))
    jcost = cost(jm, total("judge", "input_tokens"), total("judge", "cache_write_tokens"),
                 total("judge", "cache_read_tokens"), total("judge", "output_tokens"))
    print(f"cost (list-price equivalent): answer ${acost:.2f} + judge ${jcost:.2f}")
    fell_back = [r["question_id"] for r in rows if r["answer_model"] != am or r["judge_model"] != jm]
    refused = [r["question_id"] for r in rows if r["answer_stop_reason"] == "refusal"]
    if fell_back:
        print(f"served by a fallback model: {len(fell_back)} {fell_back[:5]}")
    if refused:
        print(f"refused: {len(refused)} {refused[:5]}")
    if errors:
        print(f"errored (rerun to retry): {len(errors)}")


def cmd_estimate(args) -> None:
    """Offline: how many prompt tokens a run would send, and what it would cost."""
    instances = data.stratified_sample(data.load(args.split), args.n, args.seed)
    method = make_method(args.method, args.budget)
    tokens = [
        estimate_tokens(answer_prompt(method.build(i).sessions, i.question_date, i.question)) for i in instances
    ]
    total = sum(tokens)
    pin, _ = PRICES.get(args.answer_model, (0.0, 0.0))
    print(f"{method.name} on {len(instances)} questions: ~{total / len(tokens):,.0f} prompt tokens/question, "
          f"~{total / 1e6:.2f}M total, ~${total * pin / 1e6:.2f} answer input on {args.answer_model} "
          "(estimate; newer Claude tokenizers can count up to ~1.35x more)")


def cmd_recall(args) -> None:
    """Offline: how much of the evidence a method keeps, per category. No LLM calls."""
    method = make_method(args.method, args.budget)
    instances = drop_uncovered(method, select(args, data.load(args.split)))
    if hasattr(method, "warm"):
        method.warm(instances)
    hit: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # evidence hit/total, questions fully covered/n
    for inst in tqdm(instances, leave=False):
        if inst.is_abstention:
            continue
        s = method.build(inst).stats
        h = hit[inst.category]
        h[0] += s["evidence_turns_hit"]
        h[1] += s["evidence_turns_total"]
        h[2] += s["evidence_turns_hit"] == s["evidence_turns_total"]
        h[3] += 1
    print(f"{method.name}: {'category':<28}{'turn recall':>12}{'all evidence':>14}")
    for cat in sorted(hit):
        h = hit[cat]
        print(f"{'':<{len(method.name) + 2}}{cat:<28}{h[0] / h[1]:>12.1%}{h[2] / h[3]:>14.1%}")


BENCH_METHODS = ["bm25", "userfirst-bm25-f3", "userfirst-hybrid-f3", "userfirst-time-hybrid-f3"]


def cmd_bench(args) -> None:
    """The offline recall table in the README: free, deterministic, no API key.

    For each dataset, method and budget: the share of questions whose evidence turns all fit in
    the retrieved context. LongMemEval-S is what pocket-recall was developed on; LoCoMo was never
    used for tuning and serves as the held-out check.
    """
    from . import locomo, userfirst
    from ..embed import Embedder

    userfirst._embedder = Embedder(data.DATA_DIR / "embeddings.sqlite", args.embed_model)
    datasets = {"LongMemEval-S (development set)": lambda: (data.load("s"), 0),
                "LoCoMo (held out)": locomo.load}
    out = [f"Evidence recall: share of questions whose evidence turns all fit in the context. "
           f"Embedding model `{args.embed_model}`; abstention questions excluded.\n"]
    for title, loader in datasets.items():
        if args.only and not any(o.lower() in title.lower() for o in args.only):
            continue
        instances, dropped = loader()
        instances = [i for i in instances if not i.is_abstention]
        cats = sorted({i.category for i in instances})
        counts = {c: sum(i.category == c for i in instances) for c in cats}
        out.append(f"\n### {title}: {len(instances)} questions" + (f" ({dropped} dropped: evidence ids match no turn)" if dropped else "") + "\n")
        out.append("| method | budget | " + " | ".join(f"{c} ({counts[c]})" for c in cats) + " | all |")
        out.append("|---|---:|" + "---:|" * (len(cats) + 1))
        for name in args.methods:
            for budget in args.budgets:
                method = make_method(name, budget)
                if hasattr(method, "warm"):
                    method.warm(instances)
                hit = defaultdict(int)
                for inst in tqdm(instances, desc=f"{name} {budget}", leave=False):
                    s = method.build(inst).stats
                    hit[inst.category] += s["evidence_turns_hit"] == s["evidence_turns_total"]
                cells = [f"{hit[c] / counts[c]:.1%}" for c in cats]
                out.append(f"| {name} | {budget:,} | " + " | ".join(cells) + f" | **{sum(hit.values()) / len(instances):.1%}** |")
                print(out[-1], flush=True)
    table = "\n".join(out) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table)
    print("\n" + table)


def _sign_test(better: int, worse: int) -> float:
    """Exact two-sided sign test on the questions where the two methods disagree."""
    from math import comb

    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(better, worse) + 1)) / 2**n
    return min(1.0, 2 * tail)


def cmd_compare(args) -> None:
    """Paired comparison of two runs, per category, from their per-question result files.

    Only questions answered by both methods through the same endpoint ("relay" field, or the
    endpoint URL) are paired: judges behind different endpoints grade differently.
    """
    def load(paths):
        rows = {}
        for path in paths:
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if "error" not in r and r.get("correct") is not None:
                    rows[(r["question_id"], r.get("relay") or r.get("endpoint") or "default")] = r
        return rows

    a, b = load(args.a), load(args.b)
    stats = defaultdict(lambda: [0, 0, 0, 0, 0])  # n, a correct, b correct, b better, b worse
    for key in a.keys() & b.keys():
        ra, rb = a[key], b[key]
        ca, cb = str(ra["correct"]) == "True", str(rb["correct"]) == "True"
        st = stats[rb["category"]]
        st[0] += 1; st[1] += ca; st[2] += cb; st[3] += cb and not ca; st[4] += ca and not cb
    total = [sum(v[i] for v in stats.values()) for i in range(5)]
    print(f"Paired questions: {total[0]} (A: {', '.join(p.name for p in args.a)}; B: {', '.join(p.name for p in args.b)})\n")
    print("| category | n | A | B | B better | B worse | p (sign test) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for cat, (n, ca, cb, up, down) in sorted(stats.items(), key=lambda kv: -kv[1][0]) + [("**all**", total)]:
        print(f"| {cat} | {n} | {ca / n:.1%} | {cb / n:.1%} | {up} | {down} | {_sign_test(up, down):.2g} |")


def cmd_extract(args) -> None:
    """Extract facts for every session of the selected questions (each session once, cached)."""
    from ..facts import Extractor
    from .baselines import _split_rounds
    from .userfirst import fact_store, user_text

    instances = select(args, data.load(args.split))
    store = fact_store()
    sessions = {s.session_id: s for i in instances for s in i.sessions}
    todo = [s for sid, s in sessions.items() if not store.has(sid)]
    n0, in0, out0 = store.usage()
    print(f"{len(instances)} questions, {len(sessions)} sessions, {len(todo)} to extract with {args.model}")
    extractor = Extractor(store, args.model)
    failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [
            pool.submit(extractor.extract, s.session_id, s.date, [user_text(r) for r in _split_rounds([s])])
            for s in todo
        ]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                fut.result()
            except Exception as e:  # rerun to retry
                failed += 1
                tqdm.write(f"{type(e).__name__}: {str(e)[:120]}")
    n1, in1, out1 = store.usage()
    print(f"extracted {n1 - n0} sessions ({failed} failed): {in1 - in0:,} input / {out1 - out0:,} output tokens "
          f"(API-reported), ~${cost(args.model, in1 - in0, 0, 0, out1 - out0):.2f} list price")


def cmd_inspect(args) -> None:
    """Offline: show what a method would put in the prompt for one question."""
    inst = next(i for i in data.load(args.split) if i.question_id == args.question_id)
    ctx = make_method(args.method, args.budget).build(inst)
    print(answer_prompt(ctx.sessions, inst.question_date, inst.question)[-args.chars:])
    print("\n---", json.dumps(ctx.stats))


def _load_dotenv() -> None:
    """Read KEY=value lines from the project's .env (gitignored) into the environment, overriding the shell (the shell may point at a different endpoint)."""
    env = data.DATA_DIR.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                os.environ[key.strip()] = value.strip().strip('"')


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(prog="pocket-recall eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--split", default="s", choices=list(data.SPLITS))
        sp.add_argument("--method", default="bm25", help="full | bm25 | userfirst[-bm25|-dense|-hybrid[-f<N>]]")
        sp.add_argument("--budget", type=int, default=8000, help="history token budget for retrieval methods")

    sp = sub.add_parser("download")
    sp.add_argument("--split", default="s", choices=[*data.SPLITS, "locomo"])

    sp = sub.add_parser("run")
    common(sp)
    sp.add_argument("--n", type=int, default=30, help="stratified sample size (500 = all)")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--answer-model", default="claude-opus-5-5", help="'echo' runs offline")
    sp.add_argument("--judge-model", default="claude-opus-5-5")
    sp.add_argument("--effort", default="medium")
    sp.add_argument("--answer-style", default="official", choices=["official", "enumerate"])
    sp.add_argument("--judge-effort", default="low")
    sp.add_argument("--no-fallbacks", action="store_true", help="disable server-side refusal fallbacks")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--name", help="run name (default derived from settings)")
    sp.add_argument("--categories", help="comma-separated categories to keep (after sampling)")
    sp.add_argument("--qids", help="file of question ids to keep (after sampling)")

    sp = sub.add_parser("estimate")
    common(sp)
    sp.add_argument("--n", type=int, default=30)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--answer-model", default="claude-opus-5-5")

    sp = sub.add_parser("recall")
    common(sp)
    sp.add_argument("--categories", help="comma-separated categories to keep, e.g. multi-session,temporal-reasoning")
    sp.add_argument("--qids", help="file of question ids to keep")

    sp = sub.add_parser("compare", help="paired comparison of two runs' per-question results")
    sp.add_argument("--a", type=Path, nargs="+", required=True, help="result files of method A (baseline)")
    sp.add_argument("--b", type=Path, nargs="+", required=True, help="result files of method B")

    sp = sub.add_parser("bench", help="offline evidence-recall table over LongMemEval-S and LoCoMo")
    sp.add_argument("--methods", nargs="+", default=BENCH_METHODS)
    sp.add_argument("--budgets", nargs="+", type=int, default=[2000, 8000])
    sp.add_argument("--embed-model", default=STORE_EMBED_MODEL, help="default: the store's default model")
    sp.add_argument("--only", nargs="+", help="datasets to run, e.g. locomo")
    sp.add_argument("--out", type=Path, help="also write the markdown table here")

    sp = sub.add_parser("extract", help="write-time fact extraction for the selected questions' sessions")
    sp.add_argument("--split", default="s", choices=list(data.SPLITS))
    sp.add_argument("--categories")
    sp.add_argument("--qids")
    sp.add_argument("--model", default="claude-haiku-4-5-20251001")
    sp.add_argument("--workers", type=int, default=8)

    sp = sub.add_parser("report")
    sp.add_argument("path", type=Path)

    sp = sub.add_parser("inspect")
    common(sp)
    sp.add_argument("question_id")
    sp.add_argument("--chars", type=int, default=3000)

    args = p.parse_args(argv)
    if args.cmd == "download":
        if args.split == "locomo":
            from . import locomo

            print(locomo.download())
        else:
            print(data.download(args.split))
    elif args.cmd == "bench":
        cmd_bench(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "estimate":
        cmd_estimate(args)
    elif args.cmd == "report":
        report(args.path)
    elif args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "recall":
        cmd_recall(args)
    elif args.cmd == "inspect":
        cmd_inspect(args)


if __name__ == "__main__":
    sys.exit(main())
