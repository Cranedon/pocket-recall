import json

from pocket_recall.importers import read_claude_code_session
from pocket_recall.store import MemoryStore
from pocket_recall.tooltext import strip_tool_text

BASH = ("<bash-input>ssh root@host 'docker compose up -d'</bash-input><bash-stdout> Container web Started"
        "</bash-stdout><bash-stderr></bash-stderr>")
ARTIFACT = ('<artifact-view-context artifact="12fe">\n{"context":{"mode":"edit"}}\n(The JSON line above is the'
            " viewer's state.)\n<artifact-content-authored-by-others/>\n</artifact-view-context>\n\n"
            "如果本金 10000，回测结束会有多少？")


def test_strip_tool_text_keeps_only_what_the_user_wrote():
    assert strip_tool_text(BASH) == ""
    assert strip_tool_text(ARTIFACT) == "如果本金 10000，回测结束会有多少？"
    for tool in ("<task-notification>\n<task-id>b1</task-id>\n</task-notification>",
                 '<scheduled-task name="daily">run the report</scheduled-task>',
                 "I hit my usage limit while you were working, but it has reset now. Please continue from where you left off.",
                 "在您工作期间，我达到了使用限额，但现在限额已重置。请从您中断的地方继续。",
                 "[Request interrupted by user]", "<unknown-new-tag>state</unknown-new-tag>",
                 "This session is being continued from a previous conversation."):
        assert strip_tool_text(tool) == "", tool
    assert strip_tool_text("[Request interrupted by user]\nno, use port 5433") == "no, use port 5433"
    assert strip_tool_text("<system-reminder>x</system-reminder>Keep it short") == "Keep it short"
    # the user's own words, pastes and tags inside a sentence stay
    for kept in ('<pasted_content id="f4e4">my notes</pasted_content>', "What did we decide about the <div> layout?",
                 "Rerun Bybit with the batch endpoint"):
        assert strip_tool_text(kept) == kept


def test_tool_text_is_neither_saved_nor_kept(tmp_path):
    lines = [{"type": "user", "origin": {"kind": "human"}, "timestamp": f"2026-11-01T10:0{i}:00Z",
              "message": {"role": "user", "content": c}} for i, c in enumerate([BASH, ARTIFACT, "Which port does staging use?"])]
    lines.insert(1, {"type": "assistant", "timestamp": "2026-11-01T10:00:30Z",
                     "message": {"role": "assistant", "content": [{"type": "text", "text": "The containers are up."}]}})
    t = tmp_path / "s.jsonl"
    t.write_text("\n".join(json.dumps(x) for x in lines))
    users = [m["content"] for m in read_claude_code_session(t) if m["role"] == "user"]
    assert users == ["如果本金 10000，回测结束会有多少？", "Which port does staging use?"]

    s = MemoryStore(tmp_path / "m.db", scorer="bm25")
    s.add_session([{"role": "user", "content": BASH}, {"role": "assistant", "content": "Done."},
                   {"role": "user", "content": ARTIFACT}, {"role": "user", "content": "plain question"}], session_id="old")
    assert s.clean_tool_text(dry_run=True) == (1, 1)
    assert s.clean_tool_text() == (1, 1)
    assert [m.content for m in s.get_session("old").messages if m.role == "user"] == [
        "如果本金 10000，回测结束会有多少？", "plain question"]
    assert s.clean_tool_text() == (0, 0)
    s.close()
