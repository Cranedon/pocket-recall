import asyncio

import pytest

from pocket_recall import mcp_server
from pocket_recall.store import MemoryStore

FILLER = "Sure! Here are some detailed thoughts on that. " * 40  # long, generic assistant reply


def chat(user: str) -> list[dict]:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": FILLER}]


@pytest.fixture
def bm25_store(tmp_path):
    s = MemoryStore(tmp_path / "m.db", scorer="bm25")
    yield s
    s.close()


def test_recall_finds_the_relevant_session(bm25_store):
    bm25_store.add_session(chat("Can you suggest a sourdough recipe? My starter is called Gerald."))
    bm25_store.add_session(chat("What should I pack for a hiking trip in Patagonia?"))
    r = bm25_store.recall("What is the name of my sourdough starter?")
    assert "Gerald" in r.text
    assert r.rounds_considered == 2


def test_recall_respects_the_budget(bm25_store):
    for i in range(60):
        bm25_store.add_session(chat(f"Note {i}: I tried a new tea called blend-{i} today and liked it."))
    r = bm25_store.recall("Which teas did I try?", budget_tokens=600)
    assert 0 < r.tokens_est <= 600
    # user turns are preferred over long assistant replies, so many sessions fit
    assert len(r.sessions) > 5


def test_persists_and_forgets(tmp_path):
    path = tmp_path / "m.db"
    s = MemoryStore(path, scorer="bm25")
    sid = s.add_session(chat("My cat Miso turns five in June."))
    s.close()

    s = MemoryStore(path, scorer="bm25")
    assert "Miso" in s.recall("How old is my cat?").text
    assert "Miso" in s.export_markdown()
    assert s.forget_session(sid)
    assert s.recall("How old is my cat?").sessions == []
    assert s.stats()["messages"] == 0
    s.close()


def test_append_to_session(bm25_store):
    sid = bm25_store.add_message("user", "I started learning the cello.")
    bm25_store.add_message("assistant", "Great choice!", session_id=sid)
    bm25_store.add_message("user", "My teacher is named Ilse.", session_id=sid)
    stats = bm25_store.stats()
    assert (stats["sessions"], stats["messages"]) == (1, 3)
    assert "Ilse" in bm25_store.recall("Who teaches me cello?").text


def test_hybrid_finds_implicit_mentions(tmp_path):
    """Local embeddings catch what shares no words with the question."""
    s = MemoryStore(tmp_path / "m.db")  # hybrid: BM25 + local embeddings
    s.add_session(chat("I need replacement batteries for my hearing aids, size 13."))
    for topic in ["tax deadlines", "pasta sauces", "train timetables", "guitar strings", "garden pests"]:
        s.add_session(chat(f"Tell me about {topic}."))
    r = s.recall("Which medical devices do I use?", budget_tokens=150)
    assert "hearing aids" in r.text
    s.close()


def test_mcp_tools_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("POCKET_RECALL_DB", str(tmp_path / "mcp.db"))
    monkeypatch.setattr(mcp_server, "_store", None)

    async def call(name, args):
        result = await mcp_server.server.call_tool(name, args)
        return "".join(c.text for c in result.content if c.type == "text")

    saved = asyncio.run(call("save_conversation", {"messages": chat("My favourite rice is Japanese short-grain.")}))
    assert saved.startswith("Saved 2 messages")
    recalled = asyncio.run(call("recall", {"query": "What rice do I like?", "budget_tokens": 2000}))
    assert "short-grain" in recalled and "Current date:" in recalled
    assert "1 sessions, 2 messages" in asyncio.run(call("memory_stats", {}))
    mcp_server.store().close()


def test_cjk_is_indexed_as_character_pairs():
    from pocket_recall.core import _cjk_bigrams, bm25_rank, fts_query

    assert _cjk_bigrams("我的猫 Miso") == " 我的 的猫  Miso"
    assert '"的猫"' in fts_query("我的猫多大了")
    texts = ["我家的猫叫味噌。", "推荐几款咖啡豆。"]
    assert bm25_rank(texts, "我的猫多大了")[0] == 0


def test_chinese_recall_with_default_model(tmp_path):
    from pocket_recall.core import rank
    from pocket_recall.eval.zh_check import DISTRACTORS

    s = MemoryStore(tmp_path / "m.db")  # default: multilingual embeddings
    filler = "好的！下面是一些详细的建议。" * 30
    s.add_session([{"role": "user", "content": "我对花生过敏，吃了会起疹子。"}, {"role": "assistant", "content": filler}])
    for d in DISTRACTORS:
        s.add_session([{"role": "user", "content": d}, {"role": "assistant", "content": filler}])
    question = "做饭的时候我要避开哪些食材？"
    texts = ["我对花生过敏，吃了会起疹子。"] + DISTRACTORS
    assert 0 in rank(texts, question, "hybrid", s._embedder)[:3]
    assert "花生" in s.recall(question, budget_tokens=1000).text
    s.close()


def test_time_only_question_looks_in_that_period(tmp_path):
    from datetime import datetime, timedelta

    s = MemoryStore(tmp_path / "m.db")
    now = datetime(2026, 10, 22, 20, 0).astimezone()   # Thursday; last week = Oct 12-18
    activities = ["去爬了香山", "学了做红烧肉", "看了一部纪录片", "整理了书架", "和朋友打羽毛球", "修好了自行车",
                  "去图书馆借书", "给阳台的花换盆", "参加了读书会", "去游泳馆游泳", "做了个人年度预算", "拼完了乐高"]
    for i, act in enumerate(activities * 4):
        day = now - timedelta(days=1 + i)   # one a day for ~7 weeks
        s.add_session([{"role": "user", "content": f"我今天{act}。"}, {"role": "assistant", "content": "听起来不错！"}],
                      started_at=day.isoformat())
    r = s.recall("上周我做了什么？", budget_tokens=80, now=now)
    assert r.window == (datetime(2026, 10, 12).date(), datetime(2026, 10, 18).date())
    days = {x.started_at[:10] for x in r.sessions}
    assert days and all("2026-10-12" <= d <= "2026-10-18" for d in days)
    r = s.recall("What did I do last week?", budget_tokens=80, now=now)
    assert all("2026-10-12" <= x.started_at[:10] <= "2026-10-18" for x in r.sessions)
    # With content as well, the boost is soft: last week's sessions come first, but the best
    # content match (Lego on Oct 10, just outside the window) still follows right after them.
    r = s.recall("上周我拼乐高了吗？", budget_tokens=300, now=now)
    assert "乐高" in r.text
    s.close()


@pytest.mark.parametrize("raw, prefix", [
    ("2025-03-14", "2025-03-14T00:00:00"),
    ("2025/03/14 09:30", "2025-03-14T09:30:00"),
])
def test_dates_are_normalised(raw, prefix):
    from pocket_recall.store import normalize_time

    assert normalize_time(raw).startswith(prefix)


def test_utc_times_are_stored_as_local_time():
    from datetime import datetime

    from pocket_recall.store import normalize_time

    out = normalize_time("2025-03-14T09:30:00Z")
    assert datetime.fromisoformat(out) == datetime.fromisoformat("2025-03-14T09:30:00+00:00")
    assert out[-6:] == datetime.now().astimezone().isoformat()[-6:]  # local offset


def test_mcp_keeps_original_dates_and_rejects_bad_ones(tmp_path, monkeypatch):
    monkeypatch.setenv("POCKET_RECALL_DB", str(tmp_path / "mcp.db"))
    monkeypatch.setattr(mcp_server, "_store", None)

    async def call(name, args):
        result = await mcp_server.server.call_tool(name, args)
        return "".join(c.text for c in result.content if c.type == "text")

    bad = asyncio.run(call("save_conversation", {"messages": chat("hi"), "started_at": "last tuesday"}))
    assert bad.startswith("Not saved")
    assert mcp_server.store().stats()["messages"] == 0
    asyncio.run(call("save_conversation", {"messages": chat("I ran a half marathon."), "started_at": "2025-03-14"}))
    assert "Session 2025-03-14" in asyncio.run(call("recall", {"query": "Did I run a race?"}))
    mcp_server.store().close()


def test_long_messages_are_excerpted_around_the_question():
    from pocket_recall.core import estimate_tokens, excerpt

    filler = " ".join(f"Sentence {i} is about nothing in particular." for i in range(200))
    text = "Here is my resume. " + filler + " I co-founded Tropical Services in the Caribbean. " + filler
    out = excerpt(text, "What business did I run in the Caribbean?", 120, scorer="bm25")
    assert out.startswith("Here is my resume.")
    assert "Tropical Services" in out
    assert "…" in out and estimate_tokens(out) <= 130


def test_empty_time_window_is_reported(tmp_path):
    from datetime import datetime

    s = MemoryStore(tmp_path / "m.db", scorer="bm25")
    s.add_session(chat("I repainted the fence."), started_at="2026-09-01")
    now = datetime(2026, 10, 22).astimezone()
    assert s.recall("What did I do last week?", now=now).window_records == 0
    assert s.recall("What did I do last month?", now=now).window_records == 1
    s.close()


def test_full_rounds_shrink_with_the_budget(tmp_path):
    s = MemoryStore(tmp_path / "m.db", scorer="bm25")
    for day in range(1, 61):  # ~4k tokens of user turns, more than a 2k budget holds
        s.add_session([{"role": "user", "content": f"Day {day}: my bike is a red Bianchi and I rode it to work again today."},
                       {"role": "assistant", "content": "Nice ride! " * 40}], started_at=f"2026-03-{day % 28 + 1:02d}")
    assert any(m.role == "assistant" for x in s.recall("What bike do I have?", 8000).sessions for m in x.messages)
    assert all(m.role == "user" for x in s.recall("What bike do I have?", 2000).sessions for m in x.messages)
    s.close()


def test_leftover_budget_brings_whole_replies(tmp_path):
    s = MemoryStore(tmp_path / "m.db", scorer="bm25")
    long_reply = "Here is the plan. " * 400 + "The staging port is 5433."  # ~2.8k tokens
    s.add_session([{"role": "user", "content": "ok, set up staging"}, {"role": "assistant", "content": long_reply}])
    s.add_session(chat("My bike is a red Bianchi."))
    # everything the user said fits, so the rest of the budget carries replies, whole when they fit
    assert "My bike is a red Bianchi" in s.recall("What bike do I have?", 2000).text
    assert "Sure! Here are some detailed thoughts" in s.recall("What bike do I have?", 2000).text
    assert "5433" in s.recall("Which port does staging use?", 32000).text
    assert long_reply.strip() in s.recall("Which port does staging use?", 32000).text
    s.close()


def test_facts_make_implicit_mentions_findable(tmp_path):
    """Write-time facts are extra retrieval keys; extraction is faked here (no API call)."""
    from pocket_recall.facts import Fact

    class FakeExtractor:
        def __init__(self, store):
            self.store = store

        def extract(self, session_id, date, user_messages):
            facts = [Fact(i, "The user applied for a job at Acme (job application)")
                     for i, m in enumerate(user_messages) if "cover letter" in m]
            self.store.put(session_id, "fake", facts, "", (0, 0))
            return facts

    s = MemoryStore(tmp_path / "m.db", scorer="bm25", full_rounds=0)
    for day, topic in enumerate(["pasta", "trains", "taxes", "guitars", "gardens", "football", "weather", "chess"], 1):
        s.add_session(chat(f"Tell me about {topic}."), started_at=f"2026-01-{day:02d}")
    # shares no word with the question below; dated last so default order does not favour it
    # (same-second sessions would fall back to random ids and make this flaky)
    sid = s.add_session(chat("Acme cover letter draft, sales role"), started_at="2026-01-20")
    question = "Which companies did I apply for jobs at?"
    assert "Acme" not in s.recall(question, budget_tokens=60).text   # no word overlap without facts
    result = s.extract_facts(extractor=FakeExtractor(s.facts))
    assert result["sessions"] == 9 and not result["failed"]
    assert "Acme" in s.recall(question, budget_tokens=60).text      # found through its fact
    assert s.stats()["facts"] == 1
    s.forget_session(sid)
    assert s.stats()["facts"] == 0
    s.close()


def test_answers_in_assistant_replies(tmp_path):
    """With assistant keys, a conclusion that only the assistant wrote can be recalled."""
    s = MemoryStore(tmp_path / "m.db", scorer="bm25", full_rounds=0, assistant_keys=True)
    s.add_session([{"role": "user", "content": "我换了一个中转，你帮我测一下"},
                   {"role": "assistant", "content": FILLER + " 测试发现这个中转会注入隐藏的系统提示，模型自称 Zephyr。" + FILLER}])
    for topic in ["pasta", "trains", "taxes", "guitars"]:
        s.add_session(chat(f"Tell me about {topic}."))
    r = s.recall("新中转有什么问题？模型自称什么？", budget_tokens=400)
    assert "Zephyr" in r.text
    off = MemoryStore(tmp_path / "m.db", scorer="bm25", full_rounds=0, assistant_keys=False)
    assert "Zephyr" not in off.recall("新中转有什么问题？模型自称什么？", budget_tokens=400).text
    off.close()
    s.close()


def test_cjk_text_is_estimated_conservatively():
    from pocket_recall.core import CJK_TOKENS_PER_CHAR, estimate_tokens

    english = "I moved to Lisbon in March and joined a climbing gym. " * 10
    assert estimate_tokens(english) == int(len(english) / 2.6) + 1  # unchanged for English
    chinese = "我三月份搬到了里斯本，加入了一家攀岩馆。" * 10
    cjk = sum("一" <= c <= "鿿" for c in chinese)
    assert estimate_tokens(chinese) >= cjk * CJK_TOKENS_PER_CHAR
    assert estimate_tokens("东京 Tokyo 2026") == int(len(" Tokyo 2026") / 2.6 + 2 * CJK_TOKENS_PER_CHAR) + 1


def test_chinese_recall_stays_within_budget(tmp_path):
    from pocket_recall.core import CJK_TOKENS_PER_CHAR

    s = MemoryStore(tmp_path / "m.db", scorer="bm25")
    for i in range(40):
        s.add_session(chat(f"第{i}次聊天：我在研究量化回测框架，今天试了止损单和仓位管理，结果还不错。" * 3),
                      started_at=f"2026-01-{i % 28 + 1:02d}")
    r = s.recall("我试过哪些止损和仓位管理的方法？", budget_tokens=1000)
    cjk = sum("一" <= c <= "鿿" for c in r.text)
    assert r.sessions and r.tokens_est <= 1000 and cjk * CJK_TOKENS_PER_CHAR <= 1000
    s.close()


def test_reply_lane_finds_answers_written_by_the_assistant(tmp_path):
    filler = "We talked about many unrelated things that day, like the weather and the news. " * 6

    def build(lane):
        s = MemoryStore(tmp_path / f"lane-{lane}.db", scorer="bm25", reply_lane=lane, full_rounds=0)
        for day in range(1, 25):
            s.add_session([{"role": "user", "content": f"Let's keep going on the report, part {day}. {filler}"},
                           {"role": "assistant", "content": f"Done with part {day}. {filler}"}],
                          started_at=f"2026-02-{day:02d}")
        s.add_session([{"role": "user", "content": "ok, do it"},
                       {"role": "assistant", "content": filler + " The relay injected a hidden system prompt, "
                        "so token usage jumped between 444 and 7,619 tokens. " + filler}],
                      started_at="2026-03-01")
        return s

    question = "What was wrong with the relay's token usage?"
    off, on = build(0.0), build(0.15)
    assert "7,619" not in off.recall(question, budget_tokens=4000).text
    assert "7,619" in on.recall(question, budget_tokens=4000).text
    assert "7,619" not in on.recall(question, budget_tokens=2000).text  # no lane below 4k tokens
    off.close(), on.close()
