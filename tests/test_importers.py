import json
import zipfile

from pocket_recall.importers import import_claude_code, import_claude_export, read_claude_code_session
from pocket_recall.store import MemoryStore


def _line(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False) + "\n"


def write_transcript(path):
    path.write_text("".join([
        _line(type="user", timestamp="2026-10-20T08:00:00Z", origin={"kind": "human"},
              message={"role": "user", "content": "我午饭不吃香菜<system-reminder>ignore me</system-reminder>"}),
        _line(type="assistant", timestamp="2026-10-20T08:00:05Z",
              message={"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"},
                                                         {"type": "tool_use", "name": "Bash", "input": {}}]}),
        _line(type="user", timestamp="2026-10-20T08:00:06Z",
              message={"role": "user", "content": [{"type": "tool_result", "content": "ls output"}]}),
        _line(type="assistant", timestamp="2026-10-20T08:00:09Z",
              message={"role": "assistant", "content": [{"type": "text", "text": "记住了，不放香菜。"}]}),
        _line(type="assistant", timestamp="2026-10-20T08:00:10Z",
              message={"role": "assistant", "content": [{"type": "text", "text": "还有别的吗？"}]}),
        _line(type="user", timestamp="2026-10-20T08:05:00Z", origin={"kind": "task-notification"},
              message={"role": "user", "content": "<task-notification>done</task-notification>"}),
        _line(type="user", timestamp="2026-10-20T08:06:00Z", isMeta=True,
              message={"role": "user", "content": "Base directory for this skill"}),
        _line(type="user", timestamp="2026-10-20T08:07:00Z", isSidechain=True,
              message={"role": "user", "content": "sub-agent prompt"}),
        _line(type="queue-operation", timestamp="2026-10-20T08:07:30Z"),
        "not json\n",
        _line(type="user", timestamp="2026-10-20T08:10:00Z",
              message={"role": "user", "content": "订周五去成都的机票，靠窗"}),
    ]), encoding="utf-8")


def test_reads_only_human_prompts_and_written_replies(tmp_path):
    f = tmp_path / "abcdef123456789.jsonl"
    write_transcript(f)
    msgs = read_claude_code_session(f)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == "我午饭不吃香菜"
    assert msgs[1]["content"] == "记住了，不放香菜。\n\n还有别的吗？"
    assert msgs[2]["content"] == "订周五去成都的机票，靠窗"


def test_claude_code_import_is_incremental(tmp_path):
    proj = tmp_path / "projects" / "-Users-me-foo"
    proj.mkdir(parents=True)
    write_transcript(proj / "abcdef123456789.jsonl")
    (proj / "sub").mkdir()
    write_transcript(proj / "sub" / "agent.jsonl")  # sub-agent folders are skipped
    store = MemoryStore(tmp_path / "m.db", scorer="bm25")
    assert import_claude_code(store, tmp_path / "projects")["messages"] == 3
    assert import_claude_code(store, tmp_path / "projects")["messages"] == 0  # nothing new
    with (proj / "abcdef123456789.jsonl").open("a", encoding="utf-8") as f:
        f.write(_line(type="user", timestamp="2026-10-21T09:00:00Z", origin={"kind": "human"},
                      message={"role": "user", "content": "改成周六的航班"}))
    assert import_claude_code(store, tmp_path / "projects")["messages"] == 1
    assert store.stats()["sessions"] == 1
    assert "香菜" in store.recall("我有什么忌口？", 2000).text
    store.close()


def test_claude_export_from_zip(tmp_path):
    export = [{
        "uuid": "11111111-2222-3333-4444-555555555555", "name": "travel",
        "created_at": "2026-09-01T10:00:00.000000Z",
        "chat_messages": [
            {"sender": "human", "created_at": "2026-09-01T10:00:00Z", "text": "",
             "content": [{"type": "text", "text": "帮我比较下这两个航班"}],
             "attachments": [{"file_name": "flights.txt", "extracted_content": "CA1234 07:00 / MU5678 13:00"}]},
            {"sender": "assistant", "created_at": "2026-09-01T10:00:10Z", "text": "建议选下午的 MU5678。"},
        ],
    }]
    z = tmp_path / "export.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("data/conversations.json", json.dumps(export))
    store = MemoryStore(tmp_path / "m.db", scorer="bm25")
    assert import_claude_export(store, z) == {"sessions": 1, "messages": 2}
    assert import_claude_export(store, z) == {"sessions": 0, "messages": 0}
    text = store.recall("我上次看的是哪两个航班？", 4000).text
    assert "CA1234" in text and "MU5678" in text
    store.close()


def test_hook_saves_new_turns_and_never_fails(tmp_path, monkeypatch):
    import io
    import sys

    import pytest

    from pocket_recall import hook

    f = tmp_path / "abcdef123456789.jsonl"
    write_transcript(f)
    monkeypatch.setattr(hook, "HOME", tmp_path / "home")
    monkeypatch.setenv("POCKET_RECALL_DB", str(tmp_path / "memory.db"))
    for stdin in [json.dumps({"transcript_path": str(f)}), "not json", "{}"]:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
        with pytest.raises(SystemExit) as exit_:
            hook.main()
        assert exit_.value.code == 0
    store = MemoryStore(tmp_path / "memory.db", scorer="bm25")
    assert store.stats()["messages"] == 3
    store.close()
    log = (tmp_path / "home" / "hook.log").read_text()
    assert "saved 3 messages" in log and "JSONDecodeError" in log
