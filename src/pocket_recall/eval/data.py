"""LongMemEval dataset loading and sampling."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
# In a source checkout the datasets live in ./data; an installed package keeps them in ~/.pocket-recall.
DATA_DIR = Path(os.environ.get("POCKET_RECALL_DATA_DIR") or (
    _REPO / "data" if (_REPO / "pyproject.toml").exists() else Path("~/.pocket-recall/eval-data").expanduser()))

SPLITS = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}


@dataclass
class Turn:
    role: str
    content: str
    has_answer: bool


@dataclass
class Session:
    session_id: str
    date: str
    turns: list[Turn]


@dataclass
class Instance:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: list[Session]  # in the order given by the dataset
    answer_session_ids: list[str]

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith("_abs")

    @property
    def category(self) -> str:
        """Question type, with abstention reported as its own ability."""
        return "abstention" if self.is_abstention else self.question_type


def _parse(raw: dict) -> Instance:
    sessions = [
        Session(
            session_id=sid,
            date=date,
            turns=[Turn(t["role"], t["content"], bool(t.get("has_answer", False))) for t in turns],
        )
        for sid, date, turns in zip(raw["haystack_session_ids"], raw["haystack_dates"], raw["haystack_sessions"])
    ]
    return Instance(
        question_id=raw["question_id"],
        question_type=raw["question_type"],
        question=raw["question"],
        answer=str(raw["answer"]),
        question_date=raw["question_date"],
        sessions=sessions,
        answer_session_ids=list(raw["answer_session_ids"]),
    )


def load(split: str = "s") -> list[Instance]:
    path = DATA_DIR / SPLITS[split]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `pocket-recall eval download --split {split}` first")
    with path.open() as f:
        return [_parse(raw) for raw in json.load(f)]


def stratified_sample(instances: list[Instance], n: int, seed: int = 0) -> list[Instance]:
    """Pick about n instances spread evenly over categories, deterministically."""
    if n >= len(instances):
        return instances
    by_cat: dict[str, list[Instance]] = defaultdict(list)
    for inst in instances:
        by_cat[inst.category].append(inst)
    rng = random.Random(seed)
    for group in by_cat.values():
        rng.shuffle(group)
    picked: list[Instance] = []
    cats = sorted(by_cat)
    i = 0
    while len(picked) < n:
        group = by_cat[cats[i % len(cats)]]
        idx = i // len(cats)
        if idx < len(group):
            picked.append(group[idx])
        i += 1
    order = {inst.question_id: k for k, inst in enumerate(instances)}
    return sorted(picked, key=lambda inst: order[inst.question_id])


def download(split: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download("xiaowu0162/longmemeval-cleaned", SPLITS[split], repo_type="dataset", local_dir=DATA_DIR)
    )
