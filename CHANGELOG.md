# Changelog

## 0.2.0

- **Automatic memory injection.** A new UserPromptSubmit hook asks the running MCP server (over a local socket, so the model is already loaded) for memory relevant to each prompt and adds up to 1,000 tokens to Claude's context when a relevance gate passes. The current session is excluded, a message is never injected twice in one session, and questions that only name a period ("what did I do last week") pass when that period has records. It adds 0.1–0.3 s to a prompt. In a replay of real Claude Code sessions, 40% of prompts got an injection, about 500 tokens each. Turn it off with `POCKET_RECALL_INJECT=0`.
- **Reply lane: answers the assistant wrote come back.** From 4,000 tokens of budget on, recall gives 15% of the budget to the pieces of assistant replies that match the question best, when a reply matches better than anything the user said. On held-out real-user questions whose answer is in an assistant reply, recall at 8k went from 4 to 9 of 18, with nothing lost elsewhere. `POCKET_RECALL_REPLY_LANE=0` turns it off.
- **LoCoMo split into dev and test halves** by conversation, so it can stay a held-out check while LoCoMo is used for tuning.
- `pocket-recall eval inject` measures the gate and the injected evidence on LongMemEval and LoCoMo and replays a store's sessions to show real injection frequency and cost. `demo_check --gate` checks what injection alone would answer.
- Recall is faster in a long-running server: vectors, chunks and CJK bigrams are cached in memory.
- **Topical segments as context keys** (after SeCom, ICLR 2025, and CPP, SIGDIAL 2026). Consecutive rounds about one topic form a segment: a new one starts where a user turn drifts away from the segment so far, and short acknowledgements ("ok", "改好了") never start one. Segments are ranked as a whole (BM25 over their user turns, cosine of their centroid) and fused into the round ranking, so a round whose user turn says little borrows the topic of its neighbours. No model call and no new embeddings.
- **Entity-driven second pass** (after SmartSearch, 2026). Names, numbers and code identifiers that are rare in the store, absent from the question and present in the best rounds of the first pass are added to the question for a second BM25 pass. A generic version that took any rare word drifted the query and was dropped.
- Together, on held-out data: LoCoMo test half 70.7% → 72.6% (2k) and 85.2% → 88.1% (8k), multi-hop at 8k 60.6% → 65.2%; a second set of later-written real-user questions 15 → 16 (4k) and 16 → 17 (8k) of 24, nothing lost. On the development data nothing went down. Recall at 8k on a 665k-token store takes 0.56 s instead of 0.36 s.
- **A larger budget now moves toward the full, verbatim history.** Up to 0.2's first draft, recall used only what the user said beyond the 3 best rounds, and cut every turn to 300 tokens, so it stopped improving at 8k and left most of a large budget unused (41% used at 32k). Now, once every ranked user turn is in, the leftover budget makes rounds whole (their replies, and pasted documents that were excerpted), in an order that fuses the user-turn ranking, the best piece of each reply and BM25 over the whole round. The rendered text is also checked against the budget, so recall never exceeds it (it could by 2–3%). Results at 8k and below do not change. On the real-user questions at 16k / 32k / 64k: 65 / 66 / 68 of 70 (development) and 20 / 22 / 24 of 24 (the later-written set), up from 14 at every budget above 8k. On LongMemEval every category reaches 100% at 64k.

## 0.1.2

- **Chinese, Japanese and Korean text no longer overruns the token budget.** Packing estimated every character at 2.6 per token, a ratio measured on English. CJK characters are now counted at 1.5 tokens each, a deliberately high guess, and other text keeps 2.6. Under the new estimate, 0.1.1's packs for Chinese questions were 1.8× the budget (median; up to 2.8×). Because recall now really stays inside the budget, Chinese real-user questions that came back complete went from 45 to 43 of 70 at 2k and from 60 to 57 at 8k. English results do not change. See [the evaluation report](docs/EVALUATION.md#token-estimate-for-chinese-012).
- Overlong sentences are cut by estimated tokens, not by a fixed character count.

## 0.1.1

- The importer and the Stop hook no longer store Claude Code's compaction summaries ("This session is being continued…") as user messages. A summary covers a whole session, so it outranked the real turns. To clean an existing store, delete the user messages that start with that sentence.

## 0.1.0

- First release: SQLite store with hybrid recall within a token budget, MCP server, Claude Code plugin (Stop hook, skills), secret masking on write, web workbench, and the LongMemEval / LoCoMo harness.
