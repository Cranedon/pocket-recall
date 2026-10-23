"""Turn time words in a question into a date range, relative to "now".

"上周我做了什么" / "what did I do last week" -> (start, end) dates, so retrieval can look at
conversations from that period first. Rule-based, Chinese and English. Only explicit references
to *when something happened* produce a range; "how many weeks ago did I..." asks for a duration
and produces none.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

Window = tuple[date, date]

_ZH_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
           "几": 3, "半": 0}
_EN_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "a few": 3, "few": 3}
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"]
_MONTH_RE = "|".join(_MONTHS)


def _zh_int(s: str) -> int | None:
    if s.isdigit():
        return int(s)
    if s in _ZH_NUM:
        return _ZH_NUM[s]
    if s.startswith("十") and len(s) == 2 and s[1] in _ZH_NUM:  # 十二
        return 10 + _ZH_NUM[s[1]]
    if len(s) == 2 and s.endswith("十") and s[0] in _ZH_NUM:  # 二十
        return _ZH_NUM[s[0]] * 10
    return None


def _en_int(s: str) -> int | None:
    s = s.lower().strip()
    return int(s) if s.isdigit() else _EN_NUM.get(s)


def _month_window(month: int, now: date, year: int | None = None) -> Window:
    """A named month: this year's if it has started, otherwise last year's."""
    y = year if year is not None else (now.year if month <= now.month else now.year - 1)
    start = date(y, month, 1)
    end = date(y + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return start, end


def _ago(n: int, unit: str, now: date) -> Window:
    """'N units ago' is approximate: allow slack that grows with the distance."""
    days = {"d": 1, "w": 7, "m": 30, "y": 365}[unit] * n
    slack = {"d": max(1, n // 7), "w": 4, "m": 12, "y": 60}[unit]
    centre = now - timedelta(days=days)
    return centre - timedelta(days=slack), min(now, centre + timedelta(days=slack))


def _within(n: int, unit: str, now: date) -> Window:
    days = {"d": 1, "w": 7, "m": 30, "y": 365}[unit] * n
    return now - timedelta(days=days), now


def _span(a: Window, b: Window) -> Window:
    return min(a[0], b[0]), max(a[1], b[1])


def _week(now: date, offset: int) -> Window:
    monday = now - timedelta(days=now.weekday()) + timedelta(weeks=offset)
    return monday, min(now, monday + timedelta(days=6))


def _prev_month(now: date) -> Window:
    first = now.replace(day=1)
    return _month_window((first - timedelta(days=1)).month, now, (first - timedelta(days=1)).year)


_ZH_N = r"(\d+|[一两二三四五六七八九十几半]{1,2})"
_EN_N = r"(\d+|a couple of|a few|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|few)"
_ZH_UNIT = {"天": "d", "日": "d", "周": "w", "星期": "w", "礼拜": "w", "个月": "m", "月": "m", "年": "y"}
_EN_UNIT = {"day": "d", "week": "w", "month": "m", "year": "y"}


def time_window(query: str, now: datetime | date) -> Window | None:
    """The date range a question refers to, or None when it names no point in time."""
    ref = time_reference(query, now)
    return ref[0] if ref else None


# Words that carry no topic: "上周我都聊了些什么" / "what did I do last week" are about time only.
_ZH_FILLER = re.compile(r"[我你他她们的了吗呢么吧啊呀都些也还就在有是和过跟给着得地个件次]|什么|怎么|哪些|哪里|干嘛|干啥|做|干|聊|说|讲|提到|发生|事情|事儿|一下|那天|那时|时候|期间|都|一些|有没有|告诉|回顾|总结")
_EN_FILLER = set("what did i do we you talk talked about say said mention mentioned happen happened have had was were "
                 "the a an any anything something me my in on at of to and or tell remind recap summarize "
                 "during that this those these up get got go went been doing".split())


def is_time_only(rest: str) -> bool:
    """True when nothing but filler is left once the time words are removed."""
    zh = _ZH_FILLER.sub("", re.sub(r"[\W_]+", "", rest))
    en_words = [w for w in re.findall(r"[a-z']+", rest.lower()) if w not in _EN_FILLER]
    zh_left = re.sub(r"[a-z0-9']+", "", zh.lower())
    return not zh_left and not en_words


def time_reference(query: str, now: datetime | date) -> tuple[Window, str] | None:
    """(date range, the query with the time words removed), or None."""
    today = now.date() if isinstance(now, datetime) else now
    q = query.lower()

    # Questions about durations ("how many weeks ago", "多少天前", "多久以前") ask for a number.
    # "几天前" usually means "a few days ago", a point in time, so it is not excluded.
    if re.search(r"how (many|long)\b[^?]*\b(ago|since|before|after|passed)", q) or re.search(r"(多少|多久)[天周星期个月年]*(前|以前|之前|了)", q):
        return None

    rules: list[tuple[str, callable]] = [
        (r"前天|day before yesterday", lambda m: (today - timedelta(days=2),) * 2),
        (r"昨天|昨晚|yesterday|last night", lambda m: (today - timedelta(days=1),) * 2),
        (r"今天|今早|今晚|today|this morning|tonight", lambda m: (today,) * 2),
        (r"\bthe (?:last|past) (week|month|year)\b", lambda m: _within(1, m.group(1)[0], today)),
        (r"上上周|上上个?星期|the week before last", lambda m: _week(today, -2)),
        (r"上周|上个?星期|上个?礼拜|last week", lambda m: _week(today, -1)),
        (r"这周|本周|这个?星期|this week", lambda m: _week(today, 0)),
        (r"上个月|上月|last month", lambda m: _prev_month(today)),
        (r"这个月|本月|this month", lambda m: (today.replace(day=1), today)),
        (r"去年\s*(\d{1,2}|十[一二]?|[一二三四五六七八九])\s*月",
         lambda m: _month_window(_zh_int(m.group(1)), today, today.year - 1)),
        (r"去年|last year", lambda m: (date(today.year - 1, 1, 1), date(today.year - 1, 12, 31))),
        (r"今年|this year", lambda m: (date(today.year, 1, 1), today)),
        (_ZH_N + r"\s*(个月|天|日|周|星期|礼拜|月|年)(前|以前|之前)",
         lambda m: _ago(_zh_int(m.group(1)) or 1, _ZH_UNIT[m.group(2)], today)),
        (r"(过去|最近|近)\s*" + _ZH_N + r"\s*(个月|天|日|周|星期|礼拜|月|年)",
         lambda m: _within(_zh_int(m.group(2)) or 1, _ZH_UNIT[m.group(3)], today)),
        (r"\b" + _EN_N + r"\s+(day|week|month|year)s?\s+ago",
         lambda m: _ago(_en_int(m.group(1)) or 1, _EN_UNIT[m.group(2)], today)),
        (r"\b(?:past|last)\s+" + _EN_N + r"\s+(day|week|month|year)s?",
         lambda m: _within(_en_int(m.group(1)) or 1, _EN_UNIT[m.group(2)], today)),
        (r"(\d{4})\s*年\s*(\d{1,2}|十[一二]?|[一二三四五六七八九十])\s*月",
         lambda m: _month_window(_zh_int(m.group(2)), today, int(m.group(1)))),
        (r"(\d{1,2}|十[一二]?|[一二三四五六七八九])\s*月?\s*(?:和|到|至|或|、|-)\s*(\d{1,2}|十[一二]?|[一二三四五六七八九])\s*月",
         lambda m: _span(_month_window(_zh_int(m.group(1)), today), _month_window(_zh_int(m.group(2)), today))),
        (r"(\d{1,2}|十[一二]?|[一二三四五六七八九])\s*月份?(?![\d日号])",
         lambda m: _month_window(_zh_int(m.group(1)), today)),
        (r"\b(" + _MONTH_RE + r")\s+(?:and|or|to|through|-)\s+(" + _MONTH_RE + r")\b",
         lambda m: _span(_month_window(_MONTHS.index(m.group(1)) + 1, today),
                         _month_window(_MONTHS.index(m.group(2)) + 1, today))),
        (r"\b(?:in|during|of|since)\s+(?:the month of\s+)?(" + _MONTH_RE + r")\b",
         lambda m: _month_window(_MONTHS.index(m.group(1)) + 1, today)),
    ]
    for pattern, make in rules:
        m = re.search(pattern, q)
        if m:
            window = make(m)
            if window and window[0] <= window[1]:
                return window, q[: m.start()] + " " + q[m.end():]
    return None
