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
| [LoCoMo](https://github.com/snap-research/locomo) | Held out in 0.1. From 0.2 split by conversation: the 5 with the smallest `sha1(sample_id)` are the dev half, the other 5 the test half, read once per release | 10 conversations, 1,977 questions (9 dropped: their evidence ids match no utterance); dev 801, test 730 non-abstention | 18k–37k tokens, 19–32 sessions |

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

### Token estimate for Chinese (0.1.2)

Packing estimates tokens from characters. Up to 0.1.1, every character counted as 1/2.6 of a token, a ratio measured on English LongMemEval text with Claude's tokenizer. Chinese takes far more tokens per character, so Chinese packs overran their budget. 0.1.2 counts each CJK character as 1.5 tokens, a deliberately high guess, because the relay used here offers no token-count endpoint to measure the real ratio. Other text keeps 2.6.

| On the 58 Chinese real-user questions | 2k budget | 8k budget |
|---|---:|---:|
| 0.1.1 pack size under the new estimate (median / max) | 1.84× / 2.83× | 1.68× / 2.26× |

| Real-user questions with all expected words recalled (70) | 2k | 8k |
|---|---:|---:|
| 0.1.1 | 45 | 60 |
| 0.1.2 | 43 | 57 |

All 6 changed questions are Chinese, and the English store's results did not move. The lower scores are the honest price of staying inside the budget: part of 0.1.1's Chinese results came from sending up to 2.8× the context. The Chinese ranking check (18 of 20) does not depend on the budget and did not change. If Claude's real ratio is lower than 1.5 tokens per character, Chinese packs now leave some budget unused; measuring it with an official API key would let the constant be tightened.

Experiments in section 6 were run with the 0.1.1 estimate; each of their tables compares methods under the same estimate.

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

## 8. 0.2: reply lane and automatic injection

### Discipline

- **Development data:** LongMemEval-S, the LoCoMo dev half, and the 70 real-user questions that shaped 0.1.
- **Held out:** the LoCoMo test half, and 24 new real-user questions written *before any 0.2 tuning* from recent Claude Code sessions that no earlier question used (a backend architecture project, a trading-strategy review, a video project). 18 of them are answered only in an assistant reply, 6 in a user turn. The expected words were checked to exist in the store and not to appear in the question. Both sets were run once, after every default below was fixed.

### Diagnosis: why answers in assistant replies were lost

On the 7 development questions answered in a reply, the piece holding the answer ranked **1st of 1,914 reply pieces for 2 of them** and 6th for another. Retrieval was not the main problem; packing was. 0.1's experiment mixed reply pieces into the main ranking, where they pushed out the user's own turns. On the development questions, with 0.1.2's token estimate:

| Development real-user questions (70) | 2k | 4k | 8k |
|---|---:|---:|---:|
| No reply pieces | 43 | 52 | 57 |
| Reply pieces mixed into the main ranking (0.1 experiment) | 49 | — | 59 (5 questions lost) |
| Reply lane, 10% of budget | 48 | — | 59 |
| Reply lane, 15% | 51 | — | 60 |
| Reply lane, 25% | 51 | — | 60 |
| **Reply lane 15%, only from 4k, only when a reply matches better than any user turn (default)** | 43 | 57 | 60 |

Prefixing each reply piece with the user turn of its round did not help: when the user had said something informative it lifted the answer's rank (135 → 27), but after "ok" or "use claude-sonnet-5" it pushed it down (6 → 34).

**Why the lane starts at 4k.** On a LongMemEval subset (106 questions: all 56 single-session-assistant ones and 50 from other categories), the lane cost 6 questions at 2k, 1 at 4k (multi-session) and none at 8k. LongMemEval's answers sit in user turns, so the lane's share is pure cost there. Making it open only when a reply matches better than any user turn did not fix 2k: LongMemEval's assistant writes long, descriptive replies that often match better anyway. At 1k with the injection gate, the lane lowered the real-user score from 28 to 25. So below 4k every token stays with the user's own words.

### Choosing the injection gate

The gate needs to say whether memory holds something about the prompt. Positives: a question against its own history. Negatives: the same question against another question's history (LongMemEval) or another conversation (LoCoMo dev), which has the same size and style but not the answer.

| Signal | AUC LongMemEval | AUC LoCoMo dev |
|---|---:|---:|
| best cosine | 0.84 | 0.98 |
| best cosine minus mean of top 20 | 0.79 | 0.72 |
| share of content words in the best 5 | 0.89 | 0.96 |
| **best cosine + 0.3 × content-word share (used)** | **0.89** | **0.99** |

| Threshold | LongMemEval: fires with / without memory | LoCoMo dev | Development real-user questions answered by 1k injection (39 without a gate) |
|---|---:|---:|---:|
| 0.50 | 89% / 31% | 100% / 32% | 37 |
| 0.55 | 81% / 21% | 100% / 18% | 34 |
| **0.60 (default)** | **72% / 13%** | **97% / 10%** | **32** |
| 0.65 | 63% / 7% | 94% / 3% | 28 |

The real-user questions all need memory, so they show only the cost of a high threshold. The cost of a low one shows in real prompts. With only the gate, 49% (Claude Code sessions) to 94% (a WildChat job seeker who rewrote one profile dozens of times) of prompts would get an injection, because real users keep returning to related topics. The second rule, never re-inject in a session and stay quiet when the best match is not new, brought the Claude Code replay down to 37% of prompts at about 500 tokens each.

### Held-out results

| Held-out real-user questions (24) | 2k | 4k | 8k |
|---|---:|---:|---:|
| 0.1.2 | 5 | 7 | 9 |
| **0.2** | 5 | **13** | **14** |
| ↳ answer in an assistant reply (18) | 0 → 0 | 2 → 8 | 4 → 9 |
| ↳ answer in a user turn (6) | 5 → 5 | 5 → 5 | 5 → 5 |
| Automatic injection alone (1k, gate 0.6) | 3 (0 of 18 reply answers, 3 of 6 user-turn answers) | | |

Every question that changed improved. Injection alone answers few of these, because they were chosen to be answered in replies and the lane does not open at 1k. That is the honest limit of 0.2's injection: it brings back what you said, and Claude still has to call `recall` for what it concluded.

| LoCoMo (evidence recall) | dev half (801 q) 2k / 8k | **test half, held out (730 q)** 2k / 8k |
|---|---:|---:|
| `bm25` | 60.7% / 73.2% | 62.5% / 76.6% |
| `userfirst-time-hybrid-f3` | 67.8% / 84.3% | 70.7% / 85.2% |
| `userfirst-time-lane-hybrid-f3` (0.2 default) | 67.8% / 84.3% | 70.7% / 85.2% |

The lane never opens on LoCoMo (both speakers are people, so there are no assistant replies), so the 0.2 default equals 0.1's method there. The test half behaves like the dev half.

| Injection gate (1k, threshold 0.6) | fires with memory | fires without memory | evidence complete when fired | evidence injected overall |
|---|---:|---:|---:|---:|
| LongMemEval (development, 470 q) | 71.7% | 10.9% | 75.4% | 54.0% |
| **LoCoMo test half (held out, 730 q)** | **97.3%** | **6.6%** | 62.1% | 60.4% |

On the held-out half the gate false-fires less than on the dev data it was chosen on (6.6% vs 10%).

| Replay of real sessions through the injection hook (1k, gate 0.6) | prompts | inject | tokens per prompt | tokens per injection |
|---|---:|---:|---:|---:|
| Held-out Claude Code sessions | 225 | 40% | 202 | 505 |
| Development: Claude Code sessions | 180 | 37% | 189 | 507 |
| Development: WildChat job seeker | 266 | 78% | 712 | 911 |
| Development: WildChat teacher | 164 | 77% | 611 | 796 |

Latency: the hook takes 0.1 s when nothing is injected and up to 0.3 s when it is; recall on a 665k-token store takes 0.05 s, or 0.13 s at 8k with the reply lane.

## 9. Larger budgets: toward the full history

8k was meant as a stress test of a small context, not a ceiling. Scaling the budget exposed a flaw: recall stopped improving at 8k.

| Later-written real-user questions (24) | 2k | 4k | 8k | 16k | 32k | 64k |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 first draft | 5 | 13 | 14 | 14 | 14 | 14 |
| Budget used (one question) | | | 103% | 82% | 41% | 21% |

**Why.** Real Claude Code replies are long: median 1,104 tokens, 81% over 300, none over 8k. Two rules capped what any budget could bring back:

- Beyond the 3 best rounds and the reply lane, only user turns are packed. Everything the user said in the 270k-token store totals about 13k tokens, so a larger budget sat unused.
- Every packed turn was cut to its most relevant 300 tokens. At 8k, 9 of the 10 misses had their round picked and the answer cut out by that excerpt.

**Changes**, in the order they were measured:

| Change | Development (70) 8k / 16k / 32k / 64k | Later-written (24) 8k / 16k / 32k / 64k |
|---|---:|---:|
| 0.2 first draft | 60 / not measured | 14 / 14 / 14 / 14 |
| Fill leftover budget with whole replies, user-turn order, cap budget/8 for every turn | 59 / 62 / 66 / 67 | 17 / 17 / 18 / 21 |
| Same, but pasted user documents keep the 300 cap | 58 / 65 / 66 / 67 | 16 / 16 / 18 / 21 |
| Larger cap only in the fill step, so 8k is unchanged | **60** / 65 / 66 / 67 | **14** / 16 / 18 / 21 |
| Fill also restores excerpted user documents | 60 / 65 / 66 / 67 | 14 / 16 / 19 / 21 |
| **Fill order fuses user turns, reply pieces and whole-round BM25 (final)** | 60 / 65 / 66 / **68** | 14 / **20** / **22** / **24** |

The fill order was chosen by where the answer's round lands, not by the final scores:

| Order of rounds to make whole | Real-user questions (94): answer round in top 20 / MRR | LongMemEval subset (101): top 20 / MRR |
|---|---:|---:|
| What the user said (before) | 81% / 0.49 | 95% / 0.75 |
| Best reply piece + whole-round BM25 | 96% / 0.65 | 100% / 0.82 |
| **User turns + reply pieces + whole-round BM25 (used)** | 95% / 0.60 | 100% / 0.81 |

The two-signal fusion scores slightly higher, but it ignores what the user said, and in a conversation with no assistant replies (LoCoMo) it falls back to BM25 alone. The three-signal fusion holds up in both kinds of conversation.

**Budget accounting.** The packer estimated formatting per turn, and the rendered text came out 1–3% over the budget. Recall now measures the rendered text and drops the last-picked rounds until it fits.

**On LongMemEval** (106-question subset with every single-session-assistant question), every category is at 100% by 32k except single-session-assistant at 96.4%, which reaches 100% at 64k once the fill step has room. On the full set, without the fill step, multi-session goes 92.6% (8k) → 96.7% (16k) → 100% (32k).

**Caveat.** The 24 later-written questions were inspected to diagnose this, so they are no longer held out for this change. The development questions and LongMemEval show the same trend, and nothing at 8k or below changed.

## 10. Topical segments and an entity-driven second pass

Two ideas from the literature, both kept free of model calls:

- **Topical segments** ([SeCom](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e56f394bbd4f0ec81393d767caa5a31b-Abstract-Conference.html), ICLR 2025; [CPP](https://aclanthology.org/2026.sigdial-1.55.pdf), SIGDIAL 2026: segments of a few turns beat both single turns and whole sessions). Within a session, a new segment starts when a user turn's cosine to the segment's centroid drops below 0.35, at most 10 rounds; user turns under 12 characters never start one. A segment is ranked by BM25 over its user turns fused with its centroid's cosine, and that ranking is fused into the round ranking with weight 0.3. The user-turn vectors are already cached, so no new embedding is needed.
- **Entity-driven second pass** ([SmartSearch](https://arxiv.org/abs/2603.15599), 2026: deterministic entity expansion and ranking reach 93.5% on LoCoMo). From the 5 best rounds of the first pass, up to 6 entity-like words: capitalised words in mid-sentence, tokens with digits, code identifiers, Latin terms inside Chinese. They must be absent from the question and found in at most 5% of rounds. They go into a second BM25 pass, fused with weight 0.5.

### Development sweep

| Setting | Real-user dev (94) 2k / 4k / 8k | LoCoMo dev 2k / 8k (multi-hop 8k) | LongMemEval 2k / 8k |
|---|---|---|---|
| Before | 48 / 70 / 74 | 67.8% / 84.3% (54.4%) | 57.4% / 96.8% |
| Segments 0.3, split 0.35 | 49 / 71 / 75 | 71.5% / 88.1% (60.5%) | 56.0% / 97.2% |
| Segments 1.0, split 0.35, at least 3 rounds | 51 / 70 / 73 | 75.2% / 88.8% (61.2%) | 54.5% / 96.6% |
| Second pass on any rare word, 0.5 | 52 / 70 / 75 | 67.9% / 83.4% (49.0%) | 57.7% / 96.2% |
| Second pass on entities only, 0.5 | 52 / 71 / 78 | 67.3% / 84.0% (53.7%) | 57.7% / 96.0% |
| **Segments 0.3 + entity second pass 0.5 (chosen)** | **52 / 71 / 77** | **70.3% / 87.1% (57.8%)** | **57.7% / 96.8%** |

Segments help LoCoMo most, but stronger segment weights cost LongMemEval at 2k: its sessions are single-topic, so a matching segment pushes forward rounds that are not evidence. The second pass helps real users most, where the entities are tools, URLs and numbers. Its generic form, any word rare in the store, added casual words ("wanna", "believes") and drifted the query: LoCoMo multi-hop at 8k fell to 49.0%. Restricting it to entity-like words fixed that. The chosen combination is the only one that went down on none of the three development sets.

### Held-out results (run once)

| | LoCoMo test half 2k / 8k | multi-hop 8k | open-domain 8k | Later-written real-user questions, set 2 (24) 2k / 4k / 8k / 32k |
|---|---:|---:|---:|---:|
| Before | 70.7% / 85.2% | 60.6% | 45.2% | 5 / 15 / 16 / 23 |
| **Segments + entity second pass** | **72.6% / 88.1%** | **65.2%** | **52.4%** | 5 / **16** / **17** / 23 |

Set 2 was written from sessions no earlier question used (a trading-strategy review continued after the first set, and the pocket-recall work itself), before any tuning in this round, and was moved to its own directory so no development run could read it. One of its questions changed, for the better. 19 of the 24 answers sit in assistant replies, which explains the 5 at 2k: below 4k only what the user said is packed.

**Cost.** Recall at 8k on a 665k-token store: 0.56 s instead of 0.36 s; unchanged at 2k.

## 11. Reproducing

| What | Command | Cost |
|---|---|---|
| Recall tables | `pocket-recall eval bench` | free, about 30 min the first time (embedding), minutes after |
| Paired accuracy table | `pocket-recall eval compare --a … --b …` | free, from the published runs |
| One method end to end | `pocket-recall eval run --method userfirst-hybrid-f3 --budget 8000 --n 500` | about $27 at list price with Claude Opus 5.5 |
| Inspect a prompt | `pocket-recall eval inspect --method userfirst-hybrid-f3 <question_id>` | free |
| Injection gate and evidence | `pocket-recall eval inject --datasets lme locomo-test` | free |
| Injection frequency on your own history | `pocket-recall eval inject --replay ~/.pocket-recall/memory.db` | free |
| Offline dry run | `pocket-recall eval run --answer-model echo --judge-model echo` | free |

End-to-end runs need `ANTHROPIC_API_KEY`, and `ANTHROPIC_BASE_URL` if you use a relay. They resume after interruption and record the endpoint of each answer. Use `pocket-recall eval estimate` to see the token count and cost before you run.
