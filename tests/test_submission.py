"""Tests for the answer checks (``validate_answers``): pure functions, no network and no disk."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from api_fakes import GOOD_ANSWERS, QUESTIONS

from gaia_agent.api import validate_answers


def test_validate_answers_accepts_a_good_partial_submission() -> None:
    assert validate_answers(QUESTIONS, GOOD_ANSWERS) == []


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        ([{"task_id": "nope", "submitted_answer": "x"}], "unknown task_id 'nope'"),
        ([GOOD_ANSWERS[0], GOOD_ANSWERS[0]], "task_id 'task-001' is answered more than once"),
        ([{"submitted_answer": "x"}], "answer #0 has no task_id"),
        (["task-001"], "answer #0 is not an object"),
        ([{"task_id": "task-001"}], "submitted_answer must be a string"),
        ([{"task_id": "task-001", "submitted_answer": 4}], "submitted_answer must be a string"),
        ([{"task_id": "task-001", "submitted_answer": None}], "submitted_answer must be a string"),
        ([{"task_id": "task-001", "submitted_answer": "FINAL ANSWER: 4"}], "'FINAL ANSWER' marker"),
        ([{"task_id": "task-001", "submitted_answer": "final   answer 4"}], "'FINAL ANSWER' marker"),
        ([], "no answers"),
        ("task-001", "must be a list"),
    ],
)
def test_validate_answers_reports_each_kind_of_problem(answers: Any, expected: str) -> None:
    problems = validate_answers(QUESTIONS, answers)

    assert len(problems) == 1
    assert expected in problems[0]


@pytest.mark.parametrize("text", ["4", "", "Final Fantasy VII", "answer final"])
def test_validate_answers_does_not_flag_unrelated_text(text: str) -> None:
    assert validate_answers(QUESTIONS, [{"task_id": "task-001", "submitted_answer": text}]) == []


def test_validate_answers_reports_every_problem_at_once() -> None:
    answers = [{"task_id": "x", "submitted_answer": 1}, GOOD_ANSWERS[0], GOOD_ANSWERS[0]]

    assert len(validate_answers(QUESTIONS, answers)) == 3  # unknown id, not a string, duplicate


def test_validate_answers_does_not_modify_its_inputs() -> None:
    questions, answers = list(QUESTIONS), copy.deepcopy(GOOD_ANSWERS)

    validate_answers(questions, answers)

    assert questions == QUESTIONS
    assert answers == GOOD_ANSWERS
