import io
import json
import sys
import tempfile
from pathlib import Path

import pytest

from pocket_recall import inject, mcp_server


def run_hook(monkeypatch, capsys, prompt, transcript="/x/abcdef1234567890.jsonl"):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"prompt": prompt, "transcript_path": transcript})))
    with pytest.raises(SystemExit) as exit_:
        inject.main()
    assert exit_.value.code == 0
    out = capsys.readouterr().out
    return json.loads(out)["hookSpecificOutput"]["additionalContext"] if out.strip() else None


def test_injects_relevant_memory_and_skips_the_rest(tmp_path, monkeypatch, capsys):
    # Unix socket paths are limited to ~104 characters, more than pytest's tmp_path leaves room for
    monkeypatch.setattr(inject, "RUN_DIR", Path(tempfile.mkdtemp(prefix="pr-")) / "run")
    monkeypatch.setenv("POCKET_RECALL_DB", str(tmp_path / "m.db"))
    monkeypatch.setattr(mcp_server, "_store", None)
    s = mcp_server.store()
    s.add_session([{"role": "user", "content": "I am allergic to peanuts, please keep that in mind for recipes."},
                   {"role": "assistant", "content": "Noted, no peanuts."}], session_id="old", started_at="2026-01-02")
    s.add_session([{"role": "user", "content": "The staging database runs on port 5433."}],
                  session_id=inject.session_id_for("/x/abcdef1234567890.jsonl"), started_at="2026-01-03")
    mcp_server._serve_injection()

    text = run_hook(monkeypatch, capsys, "Can you suggest a dinner recipe? Remember my peanut allergy.")
    assert text and "allergic to peanuts" in text and "relevance" in text
    assert run_hook(monkeypatch, capsys, "Explain the difference between TCP and UDP in networking.") is None
    # the current session is already in Claude's context, so it is never injected
    assert run_hook(monkeypatch, capsys, "Which port does the staging database run on?") is None
    assert run_hook(monkeypatch, capsys, "ok") is None  # too short to look anything up
    # within one session the same history is not injected twice
    other = "/x/ffffeeee1111.jsonl"
    assert run_hook(monkeypatch, capsys, "Any dessert ideas? Keep my peanut allergy in mind.", other)
    assert run_hook(monkeypatch, capsys, "And a snack for later, given the peanut allergy?", other) is None
    monkeypatch.setenv("POCKET_RECALL_INJECT", "0")
    assert run_hook(monkeypatch, capsys, "Can you suggest a dinner recipe? Remember my peanut allergy.") is None
    s.close()
    monkeypatch.setattr(mcp_server, "_store", None)


def test_gate_skips_tool_messages_and_follow_ups():
    hit = {"text": "x", "relevance": 0.8, "cos": 0.8, "own": 0.3, "rare_hits": 0}
    assert inject.passes(hit, 0.6)
    assert not inject.passes({**hit, "relevance": 0.5}, 0.6)
    assert not inject.passes({**hit, "own": 0.75}, 0.6)  # the session's own prompts match as well
    assert not inject.passes(hit, 0.6, recent=10)  # deep in a session it must share rare words
    assert inject.passes({**hit, "rare_hits": 2}, 0.6, recent=10)
    assert inject.passes({"text": "x", "relevance": 0.65}, 0.6)  # an older server: its gate
    assert inject.passes({**hit, "relevance": 0.1, "time_only": True, "window_records": 3}, 0.6)
    for p in ("<task-notification>\n<task-id>b1</task-id>", "I hit my usage limit while you were working",
              "<scheduled-task name=\"daily\">", "在您工作期间，我达到了使用限额，但现在限额已重置。"):
        assert inject.is_system_prompt(p)
    assert inject.is_system_prompt("<artifact-content-authored-by-others/>\nThe summary")
    assert not inject.is_system_prompt("What did we decide about the <div> layout?")
    assert not inject.is_system_prompt('<pasted_content id="f4e4">my notes</pasted_content>')


def test_recent_prompts_come_from_the_transcript_tail(tmp_path):
    t = tmp_path / "s.jsonl"
    lines = [{"type": "user", "message": {"role": "user", "content": f"prompt number {i} about the database"}}
             for i in range(40)]
    lines += [{"type": "user", "message": {"role": "user", "content": "<task-notification>done</task-notification>"}},
              {"type": "user", "isMeta": True, "message": {"role": "user", "content": "meta text that is long enough"}},
              {"type": "assistant", "message": {"role": "assistant", "content": "an answer"}},
              {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "x"}]}}]
    t.write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    got, whole = inject.recent_prompts(str(t))
    assert whole and len(got) == inject.RECENT_PROMPTS and got[-1] == "prompt number 39 about the database"
    assert inject.recent_prompts(str(tmp_path / "missing.jsonl")) == ([], True)


def test_recent_prompts_widen_past_tool_output(tmp_path, monkeypatch):
    monkeypatch.setattr(inject, "TAIL_READS", (3_000, 1_000_000))
    t = tmp_path / "s.jsonl"
    prompts = [{"type": "user", "message": {"role": "user", "content": f"question {i} about the relay"}} for i in range(5)]
    output = [{"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "x" * 1_000}]}}
              for _ in range(10)]
    t.write_text("\n".join(json.dumps(x) for x in prompts + output))
    got, whole = inject.recent_prompts(str(t))
    assert whole and got[-1] == "question 4 about the relay"  # found by the wider read
    monkeypatch.setattr(inject, "TAIL_READS", (3_000,))
    assert inject.recent_prompts(str(t)) == ([], False)  # only tool output read: not a new session


def test_no_server_means_no_injection(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(inject, "RUN_DIR", tmp_path / "nothing-here")
    assert run_hook(monkeypatch, capsys, "What did I say about my peanut allergy?") is None
