"""Tests for ``normalize_answer`` and decimals: integral floats, rounding to a whole number and the
number of decimal places a question asks for.

Every question and answer below is invented: no benchmark question text may live in this repository.
"""

from __future__ import annotations

import pytest

from gaia_agent.answer import normalize_answer

COUNT = "How many moons does the planet Zorblax have?"
ROUND_Q = "What is the mean orbital period of Zorblax, rounded to one decimal place?"


def test_integral_floats_keep_their_decimal_part_when_the_question_talks_about_decimals() -> None:
    assert normalize_answer(ROUND_Q, 365.0) == "365.0"
    assert normalize_answer("Give it to 2 significant digits.", 5.0) == "5.0"
    assert normalize_answer("Answer to the nearest tenth.", 5.0) == "5.0"


@pytest.mark.parametrize(
    "question",
    [
        "Round your answer to the nearest integer.",
        "Give the total, rounded to the nearest whole number.",
        "Round to the nearest ten.",
        "Report it rounded to the nearest hundred.",
        "Round the result to the nearest thousand.",
    ],
)
def test_rounding_to_a_whole_number_is_not_a_request_for_decimals(question: str) -> None:
    assert normalize_answer(question, 5.0) == "5"
    assert normalize_answer(question, "5.0") == "5.0"  # text is never reformatted, only an artefact of float is


@pytest.mark.parametrize(
    ("question", "raw", "expected"),
    [
        ("Answer to two decimal places.", 12.6, "12.60"),
        ("Answer to two decimal places.", "12.6", "12.60"),
        ("Answer to two decimal places.", 12, "12.00"),
        ("Answer to two decimal places.", 5.0, "5.00"),
        ("Give it to 2 decimal places.", "7", "7.00"),
        ("Give it to 3 decimals.", 2.5, "2.500"),
        ("Rounded to one decimal.", 4, "4.0"),
        ("Round to the nearest hundredth.", "3.5", "3.50"),
        ("Round to the nearest tenth.", "-3", "-3.0"),
        ("Answer to two decimal places.", "1,234.5", "1234.50"),
    ],
)
def test_a_requested_number_of_decimals_is_filled_with_zeros(question: str, raw: object, expected: str) -> None:
    assert normalize_answer(question, raw) == expected


@pytest.mark.parametrize(
    ("question", "raw"),
    [
        ("Answer to two decimal places.", "12.345"),  # never rounded: more decimals than asked stay as they are
        ("Answer to two decimal places.", "12.60"),
        ("Answer to two decimal places.", "about 12.6 units"),
        ("Answer to two decimal places.", "12.6%"),
        ("Answer to two decimal places.", "twelve"),
        ("Answer to two decimal places.", ""),
    ],
)
def test_decimals_are_only_filled_in_a_plain_number_and_never_rounded(question: str, raw: str) -> None:
    assert normalize_answer(question, raw) == raw


def test_a_question_that_merely_mentions_decimals_as_things_does_not_ask_for_a_format() -> None:
    assert normalize_answer("Which of the two decimals is bigger?", "0.5") == "0.5"
    assert normalize_answer("How many decimal places does the number have?", "2") == "2"


def test_a_list_is_not_padded() -> None:
    assert normalize_answer("List the values to two decimal places, comma-separated.", "1.5, 2.25") == "1.5, 2.25"


def test_without_a_request_for_decimals_a_number_is_left_alone() -> None:
    assert normalize_answer(COUNT, "12.6") == "12.6"
    assert normalize_answer(COUNT, 12.6) == "12.6"
