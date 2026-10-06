"""Checks on the answers sent to the scoring API: what is valid, and the exact payload.

Pure functions only: nothing here touches the network or the disk.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gaia_agent.api import Question

FINAL_ANSWER_MARKER = re.compile(r"final\s+answer", re.IGNORECASE)
MIN_AGENT_CODE_LENGTH = 10  # "minLength" of agent_code in the OpenAPI description of the scoring API


class InvalidSubmissionError(ValueError):
    """The answers cannot be submitted; ``problems`` lists every issue found."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__(f"{len(problems)} problem(s) found: " + "; ".join(problems))
        self.problems = tuple(problems)


def identity_problems(username: object, agent_code: object) -> list[str]:
    """Problems with who is submitting and where their code lives.

    Both must be non-blank text, and ``agent_code`` (sent stripped) at least ``MIN_AGENT_CODE_LENGTH``
    characters long: the scoring API refuses a shorter one with an HTTP 422. The API does not say what the
    text must be; the course asks for the link to the code of the agent.
    """
    fields = (("username", username), ("agent_code", agent_code))
    problems = [
        f"{label} must be a non-empty string"
        for label, value in fields
        if not isinstance(value, str) or not value.strip()
    ]
    if isinstance(agent_code, str) and agent_code.strip() and len(agent_code.strip()) < MIN_AGENT_CODE_LENGTH:
        problems.append(f"agent_code must have at least {MIN_AGENT_CODE_LENGTH} characters (the scoring API's minimum)")
    return problems


def _text_problems(task_id: str, value: object) -> list[str]:
    if not isinstance(value, str):
        return [f"answer for {task_id!r}: submitted_answer must be a string"]
    if FINAL_ANSWER_MARKER.search(value):
        return [f"answer for {task_id!r}: remove the 'FINAL ANSWER' marker from submitted_answer"]
    return []


def validate_answers(questions: list[Question], answers: list[dict[str, Any]]) -> list[str]:
    """Return human-readable problems with ``answers`` (an empty list means valid).

    Checks that every task id is known, that none is repeated, and that each
    ``submitted_answer`` is a string without a "FINAL ANSWER" marker. Answering only
    some of the questions is fine. The inputs are never modified.
    """
    if not isinstance(answers, list):
        return [f"answers must be a list, got {type(answers).__name__}"]
    if not answers:
        return ["there are no answers to submit"]
    known = {q.task_id for q in questions}
    problems: list[str] = []
    answered: list[str] = []
    for index, answer in enumerate(answers):
        task_id = answer.get("task_id") if isinstance(answer, Mapping) else None
        if not isinstance(answer, Mapping):
            problems.append(f"answer #{index} is not an object")
        elif not isinstance(task_id, str) or not task_id:
            problems.append(f"answer #{index} has no task_id")
        else:
            answered.append(task_id)
            if task_id not in known:
                problems.append(f"unknown task_id {task_id!r}")
            problems.extend(_text_problems(task_id, answer.get("submitted_answer")))
    problems.extend(f"task_id {t!r} is answered more than once" for t, count in Counter(answered).items() if count > 1)
    return problems


def build_payload(username: str, agent_code: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
    """The body of ``POST /submit``: only the fields the API expects, nothing else."""
    return {
        "username": username.strip(),
        "agent_code": agent_code.strip(),
        "answers": [{"task_id": a["task_id"], "submitted_answer": a["submitted_answer"]} for a in answers],
    }
