# Changelog

## 0.1.2

- **Chinese, Japanese and Korean text no longer overruns the token budget.** Packing estimated every character at 2.6 per token, a ratio measured on English. CJK characters are now counted at 1.5 tokens each, a deliberately high guess, and other text keeps 2.6. Under the new estimate, 0.1.1's packs for Chinese questions were 1.8× the budget (median; up to 2.8×). Because recall now really stays inside the budget, Chinese real-user questions that came back complete went from 45 to 43 of 70 at 2k and from 60 to 57 at 8k. English results do not change. See [the evaluation report](docs/EVALUATION.md#token-estimate-for-chinese-012).
- Overlong sentences are cut by estimated tokens, not by a fixed character count.

## 0.1.1

- The importer and the Stop hook no longer store Claude Code's compaction summaries ("This session is being continued…") as user messages. A summary covers a whole session, so it outranked the real turns. To clean an existing store, delete the user messages that start with that sentence.

## 0.1.0

- First release: SQLite store with hybrid recall within a token budget, MCP server, Claude Code plugin (Stop hook, skills), secret masking on write, web workbench, and the LongMemEval / LoCoMo harness.
