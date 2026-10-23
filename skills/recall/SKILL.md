---
name: recall
description: Look up what the user said in earlier Claude Code sessions with the pocket-recall memory. Use when the user refers to past work or conversations ("last time", "上次", "what did we decide about X", "the bug from last week"), asks about their own preferences, setup or history, or when a task clearly depends on context from before this session.
---

# Recall past sessions

pocket-recall stores every Claude Code session verbatim in `~/.pocket-recall/memory.db` (the plugin's Stop hook saves each turn; you never need to save the current conversation). The `recall` tool of the `pocket-recall` MCP server searches it.

## When to call `recall`

- The user points to the past: "like we did last time", "the approach we settled on", "上周那个问题".
- The answer depends on facts about the user you do not have in this session: their stack, accounts, preferences, people, plans.
- Before redoing work that may already have been done or decided in an earlier session.

Do not call it for questions about the current session or the files in front of you.

## How to call it

- Write `query` as the question you need answered, in the user's language, with the time words kept: "上周我们为什么把重排序关掉了" or "what did we decide about the cache key last month". Dates in the question narrow the search to that period.
- `budget_tokens`: 2000 for a single fact, 8000 (the default) when the answer may be spread over several sessions ("which libraries have I tried", "how many times", "everything about X").
- The result starts with the current date and the period the question refers to, then excerpts under their session dates, oldest first. Long messages are cut to the parts closest to the query, marked with "…".

## Using the result

- Answer from the excerpts and say when it was ("on 2026-10-12 you chose…"). If several sessions disagree, the latest one usually wins; say so.
- If nothing relevant came back, say you found nothing in memory rather than guessing.
- Replies the assistant wrote are recalled less reliably than the user's own words. If a decision was made in an assistant reply and does not come back, try a second query with the user's wording of the problem.
