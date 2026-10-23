# Evaluation report

This is how pocket-recall was measured, what helped, what did not, and where the numbers can mislead. Every table can be regenerated with the commands shown. The offline ones are free; the end-to-end ones call an LLM and are published as per-question result files.

## 1. What is measured

**The question.** A personal assistant has months of conversations and a context budget of a few thousand tokens. Can it put the right parts of its history into that budget?

**Evidence recall (offline).** Both benchmarks mark the turns that contain the answer. A question counts as recalled only when *all* of its evidence turns end up in the context the method builds within the budget. This is stricter than the common recall@k: a multi-session question needs every session it depends on. Abstention questions, whose answer is not in the history, are excluded. Token counts use 2.6 characters per token, measured on Claude's tokenizer (2.6–3.1), so the real prompt never exceeds the budget.

**Answer accuracy (end to end).** A reader model answers from the built context, and a judge model grades the answer against the reference. Both use the official LongMemEval prompts, copied verbatim from the [LongMemEval code](https://github.com/xiaowu0162/LongMemEval) (MIT, kept in `ref/`). As in the official script, a judge reply containing "yes" counts as correct.

**Datasets.**

| Dataset | Role | Size | History per question |
|---|---|---|---|
| [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) (cleaned) | Development set: every design choice was made on it | 500 questions, 6 abilities + abstention | 115k–177k tokens, ~50 sessions |
| [LoCoMo](https://github.com/snap-research/locomo) | Held out: never looked at while tuning | 10 conversations, 1,977 questions (9 dropped: their evidence ids match no utterance) | 18k–37k tokens, 19–32 sessions |

In LoCoMo, two people talk to each other. Every utterance becomes one turn ("Name: text"), and the dialog ids that a question cites become its evidence turns. The category names follow the evaluation code of Mem0 and Zep, which most published LoCoMo numbers use.

## 2. Methods

| Name | What goes into the budget |
|---|---|
| `full` | All sessions, by date. No retrieval. |
| `bm25` | The baseline: history cut into rounds (one user message plus the replies after it), ranked by SQLite FTS5 BM25, packed until the budget is full, then put back in date order. |
| `userfirst-bm25-f3` | The 3 best rounds (BM25 over whole rounds) with their replies; the rest of the budget holds **user turns only**, ranked by BM25. |
| `userfirst-hybrid-f3` | Same packing, with user turns ranked by reciprocal rank fusion of BM25 and dense embeddings. |
| `userfirst-time-hybrid-f3` | Adds time awareness: time expressions in the question become a date range, and turns inside it get a fused-rank boost (weight 0.5). This is what `MemoryStore.recall` does by default. |

Why user-first: in LongMemEval's multi-session and temporal questions, 575 of 576 evidence turns are things the user said. User turns average about 100 tokens and assistant replies about 670, so packing whole rounds spends most of the budget on replies.

## 3. Evidence recall

```bash
pocket-recall eval download --split s
pocket-recall eval download --split locomo
pocket-recall eval bench --out docs/results/recall.md
```

The full tables, per category and budget, are in [results/recall.md](results/recall.md). They use the store's default embedding model, `paraphrase-multilingual-MiniLM-L12-v2`. Summary:

| Method | LongMemEval-S 2k | LongMemEval-S 8k | LoCoMo 2k | LoCoMo 8k |
|---|---:|---:|---:|---:|
| `bm25` (baseline) | 36.4% | 76.2% | 61.5% | 74.8% |
| `userfirst-bm25-f3` | 56.4% | 91.3% | 59.8% | 73.3% |
| `userfirst-hybrid-f3` | 57.9% | 96.4% | 68.6% | 84.1% |
| `userfirst-time-hybrid-f3` | 57.4% | **96.8%** | 69.2% | **84.7%** |

What the held-out set shows:

- **The ranking generalises.** Hybrid with time beats raw BM25 by 7.7 points at 2k and 9.9 points at 8k on conversations it was never tuned on. Almost all of that comes from the dense embeddings: BM25 user-first → hybrid is +8.8 and +10.8.
- **User-first packing does not.** On LoCoMo, `userfirst-bm25-f3` is 1.5–1.7 points *below* the baseline. Both speakers are people, so there are no long assistant replies to leave out, and the gain seen on LongMemEval (+20 and +15 points) is specific to user–assistant chats. That is the shape of a personal assistant's history, but the gain should not be read as a general property.
- **Time awareness adds little here** (+0.6). LoCoMo questions use absolute dates ("When did Caroline go to the support group?"), and the boost only works on relative expressions.
- **Multi-hop is the weak spot:** 57.3% at 8k, against 94.8% for single-hop. A question that needs several facts from different sessions is still the hardest case.

### Choosing the embedding model

| Model | LongMemEval 8k (multi-session / temporal / preference) | Chinese check (20 pairs, right answer in top 3) |
|---|---|---|
| bge-small-en-v1.5 | 94.2% / 96.1% / 100% | 5 |
| **paraphrase-multilingual-MiniLM-L12-v2 (default)** | 92.6% / 95.3% / 100% | **18** |
| bge-small-zh-v1.5 | 86.0% / 93.7% / 83.3% | 19 |
| bge-m3 | see section 6 | 19 |

The multilingual MiniLM gives up about 1.5 points on English to be usable in Chinese. The Chinese check is in `src/pocket_recall/eval/zh_check.py`. It has only 20 pairs, so it shows a direction, not a precise number.

### Time awareness

62 LongMemEval questions contain a time expression, and all of them also ask about content. Questions with all evidence recalled, by boost weight: no time handling 56, weight 0.5 **59**, weight 1 57, weight 2 55. Strict ordering by the date range also gave 55, because it pushes relevant sessions outside the range behind irrelevant ones inside it. The weight was picked on these 62 questions, so it may be slightly overfit; LoCoMo has almost no relative time expressions, so it does not test this. Questions that ask *only* about time ("what did I do yesterday?") use strict ordering. LongMemEval has none of them, so that path is covered by unit tests only.

## 4. Answer accuracy on LongMemEval-S

Settings: reader and judge `claude-opus-5-5`, reader effort medium, judge effort low, budget 8k.

### Full context vs. retrieval (35-question stratified sample)

| | Full context | BM25 8k |
|---|---:|---:|
| Accuracy | 57.1% | 71.4% |
| Input tokens per question | 177,152 | 7,820 |
| Cost at list price | $30.59 | $1.90 |

Retrieval beats reading everything. Most of full context's extra misses were the model answering "you never mentioned it" about information that sat somewhere in 177k tokens. With 5 questions per category, this is only a direction.

### BM25 baseline, all 500 questions

74.0% overall (72.6% averaged over categories). Of the 130 wrong answers, 86 had evidence missing from the context: 49 multi-session and 26 temporal. Those two abilities were the target of the memory layer.

### pocket-recall vs. BM25, paired

Paired means both answers for a question went through the same API endpoint (see section 5). That holds for 448 of the 500 questions.

```bash
pocket-recall eval compare --a docs/results/runs/longmemeval-s-bm25-8k.jsonl \
                           --b docs/results/runs/longmemeval-s-userfirst-hybrid-f3-8k.jsonl
```

| Category | n | BM25 8k | userfirst-hybrid-f3 8k | Better | Worse | p (sign test) |
|---|---:|---:|---:|---:|---:|---:|
| temporal reasoning | 127 | 75.6% | **92.1%** | 22 | 1 | 6×10⁻⁶ |
| multi-session | 121 | 54.5% | **83.5%** | 37 | 2 | 3×10⁻⁹ |
| knowledge update | 72 | 97.2% | 100% | 2 | 0 | 0.5 |
| single-session user | 64 | 95.3% | 98.4% | 2 | 0 | 0.5 |
| preference | 30 | 80.0% | **100%** | 6 | 0 | 0.03 |
| abstention | 30 | 93.3% | 90.0% | 0 | 1 | 1 |
| single-session assistant | 4 | 100% | 100% | 0 | 0 | 1 |
| **all** | **448** | **77.9%** | **92.4%** | **69** | **4** | **2×10⁻¹⁶** |

The 52 unpaired questions are mostly single-session-assistant ones: the second endpoint's balance ran out before the BM25 run finished.

**This is not a leaderboard number.** The LongMemEval paper and most vendors use GPT-4o as reader and judge. Claude Opus 5.5 is a stronger reader, and judges differ in strictness (section 5). Numbers in this report compare retrieval methods under one fixed reader and judge.

### The errors left when evidence is complete

Of the 30 wrong hybrid answers on multi-session and temporal questions, 21 had every evidence turn in the context. They were counting and aggregation mistakes: items missed or counted twice, or things the user only planned counted as done.

To test fixes without fooling ourselves, those 248 questions were split by `sha1(question_id)` into a dev half (117) and a test half (131), and tuning happened on dev only.

| Configuration (test half, 131 questions) | Accuracy | 13 target questions right |
|---|---:|---:|
| Official prompt, effort medium (original run) | 86.3% | 0 (they were chosen as wrong) |
| Same, re-run | 84.7% | 3 |
| Effort high | 84.0% | 3 |

Neither higher effort nor an "enumerate, then count" prompt (kept as `--answer-style enumerate`) helped. Two things came out of this:

- **Run-to-run noise is about ±2 points.** Re-running the same configuration fixed 3 of the 13 target questions. So compare methods question by question, and re-run before believing a small gain.
- **8 of the 13 were wrong in all three runs, and reading them showed why.** 4 have doubtful reference answers. In one, the dates in the data give 81 days (11.6 weeks) and the reference says 15 weeks. In another, the reference counts the same pair of boots twice. 2 are ambiguous, such as where "last week" begins. Only 2 are clear model errors. The remaining errors are close to the benchmark's own label noise.

## 5. Pitfall: the API endpoint changes the score

Part of the work ran through two different third-party API relays, labelled `relay-A` and `relay-B` in the published files.

- `relay-B` injected a hidden system prompt of about 4,000 words into every request, and reported token usage inconsistently: the same request was billed anywhere from 444 to 7,619 input tokens. `relay-A` occasionally over-reported too: one 600-token judge prompt sent three times was billed 917, 1,756 and 122,459 tokens. `pocket-recall eval report` now flags requests whose billed size is far from what was sent.
- **The judges graded differently.** The same `relay-A` answers to 30 preference questions were judged correct 9 times on `relay-A` and 23 times on `relay-B`. Re-judging 60 hybrid answers on both endpoints gave 54 vs 55, so the difference is concentrated in the open-ended preference rubric.

So scores from different endpoints cannot be pooled. A mixed 500-question hybrid score (92.8%) is not reported as a result, and the paired comparison above only pairs answers from the same endpoint. Within each relay: on `relay-A` (280 questions), BM25 got 68.6% and hybrid 89.3% (61 better, 3 worse). On `relay-B` (168 questions), BM25 got 93.5% and hybrid 97.6% (8 better, 1 worse); the lenient judge leaves little room there.

## 6. Experiments that did not make the default

### Write-time fact extraction

An LLM reads each session's user messages once and writes short, self-contained facts ("The user owns a kayak (outdoor gear)"). Each fact becomes an extra retrieval key for the round it came from; the prompt still gets the original words.

- **LongMemEval pilot** (32 questions: all 12 multi-session/temporal questions that hybrid still missed, plus 20 random recalled ones; 1,473 sessions, claude-haiku-4-5, $2.87). It fully recovered 4 of the 12 misses and partly recovered 1, and broke none of the 20.
- **Real users** (288 sessions from three people in [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M); claude-sonnet-5; 2,087 facts; $4.78, about 3× the expected cost because of the relay's injected prompt). On 36 questions written from their conversations, the score stayed at 30/36 (2k) and 35/36 (8k). The facts recovered one question and broke another. A job seeker had rewritten their profile dozens of times, so near-identical facts piled up and pushed out the session that mattered. Merging identical facts traded one miss for another.

**Verdict:** no measurable gain on real users, at a real cost per session. Kept as the optional `pocket-recall facts` command.

### Cross-encoder reranker

`bge-reranker-v2-m3-int8` re-orders the top 30 candidates. The Chinese check went from 18 to 20; real-user questions did not change (30/36 and 35/36); recall became about 20× slower. Reranking only 15 candidates over 600 characters made it faster, but then a fixed question failed again. Off by default (`POCKET_RECALL_RERANK_MODEL`).

### Assistant replies as retrieval keys

In coding sessions many decisions are written in the assistant's replies, which user-first packing deprioritises. Indexing 120-token chunks of replies as extra keys gave:

| | Off | On |
|---|---:|---:|
| Real-user questions, 2k / 8k (74 at the time) | 45 / **60** | 48 / 55 |
| LongMemEval subset (106 questions incl. all 56 single-session-assistant) | assistant 96.4%, others 100% | assistant 96.4%, temporal 100% → 88.9% |
| Time for those 106 questions | 14 s | 72 min |

There were two causes. Chunks lose their context: the chunk that mentions the relay's hidden prompt never says "relay", and ranked 162nd of 1,102 keys. And the small embedding model over-scores very short messages ("fixed it", "how's it going"). Off by default (`POCKET_RECALL_ASSISTANT_KEYS=1`).

### bge-m3 instead of MiniLM

Measured on an M3 MacBook (16 GB), on 500 real user messages and the same questions:

| | MiniLM (default) | bge-m3 |
|---|---:|---:|
| Embedding 500 messages | ~5 s | ~140 s |
| One query | 4 ms | 120 ms |
| Peak memory | ~1.0 GB | ~1.8 GB |
| Model size | 0.22 GB | 2.1 GB |
| Chinese check | 18 | 19 |
| 70 real-user questions, 2k / 8k | 45 / **60** | **48** / 59 |
| 55 hard LongMemEval questions (MiniLM's 15 misses + 40 random hits), all evidence | 40 | **44** (4 fixed, none broken) |

bge-m3 reads up to 8192 tokens. Batches are padded to their longest text, so for this model batching was twice as slow as embedding one passage at a time, and passages are capped at 2,000 characters. On real users it is a wash, at 25× the embedding time. Optional: `POCKET_RECALL_EMBED_MODEL=BAAI/bge-m3`.

## 7. Real-user checks

Benchmarks are synthetic, so development also used real conversations:

- three people from WildChat-1M (ODC-BY): a kindergarten teacher writing daily class notes, an account manager handling customers and travel, a job seeker rewriting a resume;
- a programmer who mixes code with everyday questions;
- the author's own Claude Code sessions.

The question sets were written by reading the conversations and checked by whether the expected words come back (`src/pocket_recall/eval/demo_check.py`). They found real bugs:

- A pasted 3,000-token resume used to crowd everything else out of a 2k budget. Long turns are now excerpted, which took the 2k score from 26/36 to 30/36.
- Full-round slots used to go to weakly related short rounds.
- In Chinese, "什么" (what) matched everywhere.
- Claude Code timestamps arrived in UTC and are now stored as local time.

Because the questions shaped the code, they are a development aid, not a benchmark, and the data is not redistributed. Users with trauma, illness or legal matters were excluded when choosing people.

## 8. Reproducing

| What | Command | Cost |
|---|---|---|
| Recall tables | `pocket-recall eval bench` | free, about 30 min the first time (embedding), minutes after |
| Paired accuracy table | `pocket-recall eval compare --a … --b …` | free, from the published runs |
| One method end to end | `pocket-recall eval run --method userfirst-hybrid-f3 --budget 8000 --n 500` | about $27 at list price with Claude Opus 5.5 |
| Inspect a prompt | `pocket-recall eval inspect --method userfirst-hybrid-f3 <question_id>` | free |
| Offline dry run | `pocket-recall eval run --answer-model echo --judge-model echo` | free |

End-to-end runs need `ANTHROPIC_API_KEY`, and `ANTHROPIC_BASE_URL` if you use a relay. They resume after interruption and record the endpoint of each answer. Use `pocket-recall eval estimate` to see the token count and cost before you run.
