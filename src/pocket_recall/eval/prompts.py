"""Answer and judge prompts, taken verbatim from the official LongMemEval code.

Source: https://github.com/xiaowu0162/LongMemEval (MIT License, Copyright (c) 2024 Di Wu)
  - answer template: src/generation/run_generation.py (history_format=nl, cot=true)
  - judge templates: src/evaluation/evaluate_qa.py::get_anscheck_prompt
Keeping these identical is what makes our numbers comparable to published ones.
"""

from __future__ import annotations

from .data import Session

ANSWER_TEMPLATE = (
    "I will give you several history chats between you and a user. Please answer the question based on the "
    "relevant chat history. Answer the question step by step: first extract all the relevant information, and "
    "then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\n"
    "Question: {}\nAnswer (step by step):"
)


def format_history(sessions: list[Session]) -> str:
    """Render sessions the way the official script does (nl format, both roles)."""
    out = ""
    for i, sess in enumerate(sessions):
        body = "".join(f"\n\n{t.role}: {t.content.strip()}" for t in sess.turns)
        out += f"\n### Session {i + 1}:\nSession Date: {sess.date}\nSession Content:\n{body}\n"
    return out


# Ours, not official: makes the model build a checked list before counting or summing.
ENUMERATE_TEMPLATE = (
    "I will give you excerpts from several history chats between you and a user, each with its date. Please answer "
    "the question based on the relevant chat history.\n\n"
    "Work in three steps.\n"
    "1. List every piece of the history that could matter for the question, each with its session date and a short "
    "quote.\n"
    "2. Check each item: Did it actually happen to the user, or is it currently true for them, rather than something "
    "planned, hypothetical, asked about in general, or about someone else? Does it fall in the time range the question "
    "asks about, counted from the current date? Is it the same thing as an item already listed, mentioned again in "
    "another chat? If a later chat updates an earlier one, keep the latest. Drop items that fail a check, and say why.\n"
    "3. Compute the answer from the items that remain and state it on the last line. If the history does not contain "
    "the information needed, say so.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:"
)

ANSWER_STYLES = {"official": ANSWER_TEMPLATE, "enumerate": ENUMERATE_TEMPLATE}


def answer_prompt(sessions: list[Session], question_date: str, question: str, style: str = "official") -> str:
    return ANSWER_STYLES[style].format(format_history(sessions), question_date, question)


_STANDARD = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
    "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
    "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
    "only contains a subset of the information required by the answer, answer no. "
)
_TAIL = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

JUDGE_TEMPLATES = {
    "single-session-user": _STANDARD + _TAIL,
    "single-session-assistant": _STANDARD + _TAIL,
    "multi-session": _STANDARD + _TAIL,
    "temporal-reasoning": _STANDARD
    + "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number "
    "of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is "
    "18), the model's response is still correct. "
    + _TAIL,
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes "
    "if the response contains the correct answer. Otherwise, answer no. If the response contains some previous "
    "information along with an updated answer, the response should be considered as correct as long as the updated "
    "answer is the required answer." + _TAIL,
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
    "The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only.",
}

ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the "
    "model correctly identifies the question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\n"
    "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)


def judge_prompt(question_type: str, question: str, answer: str, response: str, abstention: bool) -> str:
    template = ABSTENTION_TEMPLATE if abstention else JUDGE_TEMPLATES[question_type]
    return template.format(question, answer, response)


def parse_judgement(text: str) -> bool:
    # Same rule as the official script: any "yes" in the reply counts as correct.
    return "yes" in text.strip().lower()
