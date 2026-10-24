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


def test_no_server_means_no_injection(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(inject, "RUN_DIR", tmp_path / "nothing-here")
    assert run_hook(monkeypatch, capsys, "What did I say about my peanut allergy?") is None
