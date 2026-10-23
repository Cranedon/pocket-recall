"""LoCoMo (Maharana et al., 2024) as a held-out recall benchmark.

Nothing in pocket-recall was tuned on LoCoMo, so it checks whether the LongMemEval numbers
carry over. Each of its 10 conversations is between two people; every utterance becomes one
retrievable turn ("Name: text"), and the dialog ids a question cites as evidence mark the turns
that must come back. Category 5 (adversarial, no answer in the conversation) counts as
abstention and is left out of recall, as in LongMemEval.

The data (CC BY-NC 4.0) is downloaded, not redistributed: `pocket-recall eval download --split locomo`.
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path

from .data import DATA_DIR, Instance, Session, Turn

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
FILE = "locomo10.json"
# Names follow the evaluation code of Mem0 and Zep, which most published LoCoMo numbers use.
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def download() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / FILE
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    return path


def _date(text: str) -> str:
    """'1:56 pm on 8 May, 2023' -> '2023/05/08 (Mon) 13:56', the LongMemEval date format."""
    when = datetime.strptime(text.strip(), "%I:%M %p on %d %B, %Y")
    return when.strftime("%Y/%m/%d (%a) %H:%M")


def load() -> tuple[list[Instance], int]:
    """All questions, plus how many were dropped because their evidence ids match no turn."""
    path = DATA_DIR / FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `pocket-recall eval download --split locomo` first")
    instances, dropped = [], 0
    for conv in json.loads(path.read_text()):
        c = conv["conversation"]
        numbers = sorted(int(k.split("_")[1]) for k in c if re.fullmatch(r"session_\d+", k) and c[k])
        sessions_raw = [(f"{conv['sample_id']}-s{n}", _date(c[f"session_{n}_date_time"]), c[f"session_{n}"])
                        for n in numbers]
        all_ids = {t["dia_id"] for _, _, turns in sessions_raw for t in turns}
        last_date = sessions_raw[-1][1]
        for qi, qa in enumerate(conv["qa"]):
            evidence = {e for item in qa.get("evidence", []) for e in re.split(r"[;,\s]+", item) if e}
            category = qa["category"]
            if category != 5 and (not evidence or not evidence <= all_ids):
                dropped += 1
                continue
            sessions = [
                Session(sid, date, [Turn("user", f"{t['speaker']}: {t['text']}", t["dia_id"] in evidence)
                                    for t in turns])
                for sid, date, turns in sessions_raw
            ]
            instances.append(Instance(
                question_id=f"{conv['sample_id']}-q{qi}" + ("_abs" if category == 5 else ""),
                question_type=CATEGORIES[category],
                question=qa["question"],
                answer=str(qa.get("answer", qa.get("adversarial_answer", ""))),
                question_date=last_date,
                sessions=sessions,
                answer_session_ids=[s.session_id for s in sessions if any(t.has_answer for t in s.turns)],
            ))
    return instances, dropped
