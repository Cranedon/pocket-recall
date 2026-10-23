from datetime import date

import pytest

from pocket_recall.timeparse import time_window

NOW = date(2026, 10, 22)  # a Thursday


@pytest.mark.parametrize("query, expected", [
    ("昨天我吃了什么？", (date(2026, 10, 21), date(2026, 10, 21))),
    ("上周我都聊了些什么", (date(2026, 10, 12), date(2026, 10, 18))),
    ("这周我有什么安排", (date(2026, 10, 19), date(2026, 10, 22))),
    ("上个月我去了哪里", (date(2026, 9, 1), date(2026, 9, 30))),
    ("三月份我说过哪些计划", (date(2026, 3, 1), date(2026, 3, 31))),
    ("12月我在做什么", (date(2025, 12, 1), date(2025, 12, 31))),   # December hasn't come yet this year
    ("去年三月我在哪", (date(2025, 3, 1), date(2025, 3, 31))),
    ("2024年5月发生了什么", (date(2024, 5, 1), date(2024, 5, 31))),
    ("两周前我买了什么", (date(2026, 10, 4), date(2026, 10, 12))),
    ("最近三天我吃了什么", (date(2026, 10, 19), date(2026, 10, 22))),
    ("What did I do last week?", (date(2026, 10, 12), date(2026, 10, 18))),
    ("What did I buy two weeks ago?", (date(2026, 10, 4), date(2026, 10, 12))),
    ("How many museums did I visit in February?", (date(2026, 2, 1), date(2026, 2, 28))),
    ("How many times did I bake in the past two weeks?", (date(2026, 10, 8), date(2026, 10, 22))),
    ("What did I mention yesterday?", (date(2026, 10, 21), date(2026, 10, 21))),
    ("Which airline did I fly most in March and April?", (date(2026, 3, 1), date(2026, 4, 30))),
    ("三月和四月我坐了几次飞机", (date(2026, 3, 1), date(2026, 4, 30))),
    ("How many plants did I get in the last month?", (date(2026, 9, 22), date(2026, 10, 22))),
    ("What did I do last month?", (date(2026, 9, 1), date(2026, 9, 30))),
])
def test_time_references(query, expected):
    assert time_window(query, NOW) == expected


@pytest.mark.parametrize("query", [
    "我最喜欢什么米？",
    "How many weeks ago did I attend the festival?",   # asks for a duration
    "How many days had passed since I recovered from the flu?",
    "我多少天前买的烤箱？",
    "What is the order of the museums I visited?",
    "3月5日那天我在干嘛",  # a specific day, not the whole month: not handled yet
])
def test_no_time_reference(query):
    assert time_window(query, NOW) is None


@pytest.mark.parametrize("query, time_only", [
    ("上周我都聊了些什么？", True),
    ("What did I do last week?", True),
    ("昨天发生了什么事情", True),
    ("上周我拼乐高了吗？", False),
    ("How many museums did I visit in February?", False),
    ("三月份我说过哪些计划", False),
])
def test_time_only_questions(query, time_only):
    from pocket_recall.timeparse import is_time_only, time_reference

    assert is_time_only(time_reference(query, NOW)[1]) is time_only
