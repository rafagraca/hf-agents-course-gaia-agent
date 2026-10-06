"""Tests for ``normalize_answer``: a conservative clean-up of the agent's raw answer.

Every question and answer below is invented (fictional places and people): no
benchmark question text may live in this repository.
"""

from __future__ import annotations

import re

import pytest

from gaia_agent.answer import normalize_answer

PLAIN = "What is the capital of Freedonia?"
COUNT = "How many moons does the planet Zorblax have?"
LIST_Q = "Which planets does the starship Albatross visit? Answer with a comma-separated list."
TIGHT_LIST_Q = "Name the Ruritanian rivers as a comma-separated list with no spaces."
ROUND_Q = "What is the mean orbital period of Zorblax, rounded to one decimal place?"
KEEP_COMMAS_Q = "What is the population of Ruritania? Write it with a thousands separator."

UNTOUCHED = [
    "Paris",
    "paris",
    "PARIS",
    "Saint-Étienne",
    "New York City",
    "3.14159",
    "2.50",
    "007",
    "1e5",
    "five",
    "Yes",
    "x^2 + 3x",
    "O'Brien",
    "St. Louis",
    "example.com",
    "50%",
    "$5",
    "5 km",
    "Henry VIII",
    "a, b, c",
    "a;b",
    "[0, 5]",
    "(1, 2)",
    "line one\nline two",
    "Answer to the Ultimate Question",
    "Final answers matter",
    "Answering: Paris",
    "Unanswered: yes",
    "My answer: Paris",
]
NON_STRINGS = [
    (42, "42"),
    (-7, "-7"),
    (0, "0"),
    (3.5, "3.5"),
    (True, "True"),
    (False, "False"),
    (None, ""),
    (b"Paris", "Paris"),
    ([], ""),
    (["a", "b"], "a, b"),
    (("a", "b"), "a, b"),
    ({"b", "a"}, "a, b"),
    ([1, 2.0, "x"], "1, 2, x"),
    ([["a", "b"], ["c"]], "a, b, c"),
]
LABELS = [
    ("FINAL ANSWER: Paris", "Paris"),
    ("Final answer: Paris", "Paris"),
    ("final answer:Paris", "Paris"),
    ("Answer: Paris", "Paris"),
    ("ANSWER: 42", "42"),
    ("  FINAL ANSWER: Paris  ", "Paris"),
    ("**FINAL ANSWER:** Paris", "Paris"),
    ("**Final Answer**: Paris", "Paris"),
    ("The final answer is: Paris", "Paris"),
    ("The final answer is Paris", "Paris"),
    ("I compared both tables.\nFINAL ANSWER: Paris", "Paris"),
    ("FINAL ANSWER: Lyon\nFINAL ANSWER: Paris", "Paris"),
    ("FINAL ANSWER:\nParis", "Paris"),
    ("FINAL ANSWER:", ""),
    ("Answer: Answer: Paris", "Paris"),
    ("Final answer is Paris", "Paris"),
    ("I think the final answer is Paris", "Paris"),
    ("Final Answer 42", "42"),
    ("FINAL ANSWER - Paris", "Paris"),
    ("FINAL ANSWER \u2013 Paris", "Paris"),
    ("**Final answer is** Paris", "Paris"),
    ('"FINAL ANSWER: Paris"', "Paris"),
]
WRAPPERS = [
    ('"Paris"', "Paris"),
    ("'Paris'", "Paris"),
    ("`Paris`", "Paris"),
    ("\u201cParis\u201d", "Paris"),
    ("\u2018Paris\u2019", "Paris"),
    ("\u00abParis\u00bb", "Paris"),
    ('"New York"', "New York"),
    ("**Paris**", "Paris"),
    ("```\nParis\n```", "Paris"),
    ("```text\nParis\n```", "Paris"),
    ("  `Paris`  ", "Paris"),
    ('"`Paris`"', "Paris"),
    ('""', ""),
]
KEPT_QUOTES = [
    '"Paris',
    'Paris"',
    "'Tis the season",
    "Jones'",
    "It's",
    '"Say "hi" now"',
    "'It's fine'",
    '"a" and "b"',
    "**Paris",
    "Paris**",
]
WHITESPACE = [
    ("  Paris  ", "Paris"),
    ("\nParis\n", "Paris"),
    ("Paris\r\n", "Paris"),
    ("New\u00a0York", "New York"),
    ("New\u202fYork", "New York"),
    ("Par\u200bis", "Paris"),
    ("\ufeffParis", "Paris"),
    ("New   York", "New York"),
]
PERIODS = [
    ("Paris.", "Paris"),
    ("Paris. ", "Paris"),
    ("42.", "42"),
    ("3.", "3"),
    ("3.14.", "3.14"),
    ("The capital is Paris.", "The capital is Paris"),
    ('"Paris."', "Paris"),
    ("**Paris.**", "Paris"),
    ("Yes.", "Yes"),
    ("No.", "No"),
    ("1,234.", "1234"),
]
KEPT_PERIODS = [
    "Washington, D.C.",
    "Martin Luther King Jr.",
    "Acme Inc.",
    "Smith et al.",
    "John F.",
    "U.S.",
    "Ph.D.",
    "a.m.",
    "Wait...",
    "...",
    ".",
    "Paris..",
    "etc.",
    "Dr.",
    "Hudson Bay Co.",
]
THOUSANDS = [
    ("1,234", "1234"),
    ("12,345,678", "12345678"),
    ("1,234.56", "1234.56"),
    ("-1,234", "-1234"),
    ("+1,234", "+1234"),
    ("1,000", "1000"),
    ("1 234", "1234"),
    ("1 234 567", "1234567"),
    ("12 345", "12345"),
    ("1\u00a0234", "1234"),
    ("1\u202f234\u202f567", "1234567"),
]
NOT_THOUSANDS = [
    "1,23",
    "1,2345",
    "12,34,567",
    "0,123",
    "01,234",
    "1234,567",
    "1.234.567",
    "3,5",
    "1,234,56",
    "1 23",
    "1,234 people",
    "$1,234",
    "1, 234",
    "A1,234",
    "1,\u0663\u0664\u0665",
]
LIST_CASES = [
    ("a,b,c", "a, b, c"),
    ("a , b ,c", "a, b, c"),
    ("a;b;c", "a, b, c"),
    ("a\nb\nc", "a, b, c"),
    ("a, b, c", "a, b, c"),
    ("[a, b, c]", "a, b, c"),
    ("['a', 'b']", "a, b"),
    ('"a", "b"', "a, b"),
    ("red, green, blue.", "red, green, blue"),
    ("a,, b,", "a, b"),
    ("Paris (France, Europe), Rome", "Paris (France, Europe), Rome"),
    ("1 234, 5 678", "1234, 5678"),
    ("FINAL ANSWER: a,b", "a, b"),
    ("apple", "apple"),
    ("apple pie, banana split", "apple pie, banana split"),
    ("Zeta,alpha,Beta", "Zeta, alpha, Beta"),
    ("1,234", "1, 234"),
]
LIST_PHRASES = [
    "Give a comma-separated list.",
    "Answer as a comma separated list.",
    "List them separated by commas.",
    "Separate the names with commas.",
    "Items separated by a comma, please.",
    "Provide a comma-delimited list.",
    "Reply COMMA-SEPARATED.",
    "Return the values delimited by commas.",
]
NOT_LIST_PHRASES = [
    "Which list is longer?",
    "List the planets.",
    "Use semicolons to separate the names.",
    "How many commas does the poem contain?",
    "Give the digits separated by spaces.",
]
INVENTED = [
    (PLAIN, "FINAL ANSWER: Fredville.", "Fredville"),
    (COUNT, "1,024", "1024"),
    (COUNT, 12.0, "12"),
    ("What is the first name of the Ruritanian king?", '"Rudolf".', "Rudolf"),
    ("Is Freedonia landlocked? Answer yes or no.", "No.", "No"),
    (LIST_Q, "Zorblax,Quux, Gondor.", "Zorblax, Quux, Gondor"),
    (LIST_Q, ["Zorblax", "Quux"], "Zorblax, Quux"),
    (TIGHT_LIST_Q, "Elbe, Vlt", "Elbe,Vlt"),
    (TIGHT_LIST_Q, ["Elbe", "Vlt"], "Elbe,Vlt"),
    (ROUND_Q, 365.0, "365.0"),
    (KEEP_COMMAS_Q, "3,141,592", "3,141,592"),
    ("Who wrote the Ruritanian anthem?", "  Anna Quux wrote it.  ", "Anna Quux wrote it"),
    ("What is the Zorblaxian word for hello?", "`glorp`", "glorp"),
]


@pytest.mark.parametrize("raw", UNTOUCHED + KEPT_QUOTES + KEPT_PERIODS + NOT_THOUSANDS)
def test_clean_answers_are_returned_unchanged(raw: str) -> None:
    assert normalize_answer(PLAIN, raw) == raw


@pytest.mark.parametrize(("raw", "expected"), NON_STRINGS)
def test_non_string_answers_become_strings(raw: object, expected: str) -> None:
    assert normalize_answer(PLAIN, raw) == expected


def test_arbitrary_objects_use_their_str() -> None:
    class Thing:
        def __str__(self) -> str:
            return "thing"

    assert normalize_answer(PLAIN, Thing()) == "thing"


@pytest.mark.parametrize("raw", [None, 1, 2.0, b"x", "", [], {}, object(), float("nan"), float("inf"), "\x00"])
def test_the_result_is_always_a_string(raw: object) -> None:
    assert isinstance(normalize_answer(PLAIN, raw), str)


@pytest.mark.parametrize(("raw", "expected"), [(5.0, "5"), (-3.0, "-3"), (0.0, "0"), (21.0, "21")])
def test_integral_floats_lose_the_artificial_decimal_part(raw: float, expected: str) -> None:
    assert normalize_answer(COUNT, raw) == expected


@pytest.mark.parametrize("raw", [3.14159, 0.1 + 0.2, 1e20, 1e-05, float("nan"), float("inf")])
def test_other_floats_are_not_rounded_or_reformatted(raw: float) -> None:
    assert normalize_answer(COUNT, raw) == str(raw)


@pytest.mark.parametrize(("raw", "expected"), LABELS)
def test_answer_labels_are_removed(raw: str, expected: str) -> None:
    assert normalize_answer(PLAIN, raw) == expected


@pytest.mark.parametrize("raw", [raw for raw, _ in LABELS])
def test_no_final_answer_marker_survives_a_label(raw: str) -> None:
    """Submission validation rejects any "final answer" text, so a label must never be left behind."""
    assert re.search(r"final\s+answer", normalize_answer(PLAIN, raw), re.IGNORECASE) is None


@pytest.mark.parametrize(("raw", "expected"), WRAPPERS)
def test_wrapping_quotes_backticks_and_bold_are_removed(raw: str, expected: str) -> None:
    assert normalize_answer(PLAIN, raw) == expected


@pytest.mark.parametrize(("raw", "expected"), WHITESPACE)
def test_whitespace_is_cleaned(raw: str, expected: str) -> None:
    assert normalize_answer(PLAIN, raw) == expected


@pytest.mark.parametrize(("raw", "expected"), PERIODS)
def test_a_loose_final_period_is_removed(raw: str, expected: str) -> None:
    assert normalize_answer(PLAIN, raw) == expected


@pytest.mark.parametrize(("raw", "expected"), THOUSANDS)
def test_thousands_separators_are_removed_from_numbers(raw: str, expected: str) -> None:
    assert normalize_answer(COUNT, raw) == expected


@pytest.mark.parametrize(
    "question",
    [
        KEEP_COMMAS_Q,
        "Report the count using digit grouping.",
        "Give the number with commas as thousands separators.",
        "Use a thousand separator.",
        "Format it with commas.",
        "Use commas to separate thousands.",
        "Separate thousands with a comma.",
    ],
)
def test_thousands_separators_stay_when_the_question_asks_for_them(question: str) -> None:
    assert normalize_answer(question, "1,234,567") == "1,234,567"
    assert normalize_answer(question, "1 234 567") == "1 234 567"


def test_a_thousands_format_request_wins_over_a_list_request() -> None:
    question = "Give the populations as a comma-separated list, with thousands separators."

    assert normalize_answer(question, "1,234, 5,678") == "1,234, 5,678"


def test_space_grouped_digits_stay_when_the_question_asks_for_space_separated_values() -> None:
    question = "Give two numbers separated by spaces."

    assert normalize_answer(question, "100 200") == "100 200"
    assert normalize_answer(COUNT, "100 200") == "100200"


@pytest.mark.parametrize(("raw", "expected"), LIST_CASES)
def test_comma_separated_lists_get_normalised_separators(raw: str, expected: str) -> None:
    assert normalize_answer(LIST_Q, raw) == expected


@pytest.mark.parametrize("question", LIST_PHRASES)
def test_list_requests_are_recognised(question: str) -> None:
    assert normalize_answer(question, "a,b;c") == "a, b, c"


@pytest.mark.parametrize("question", NOT_LIST_PHRASES)
def test_lists_are_left_alone_unless_a_comma_separated_list_is_requested(question: str) -> None:
    assert normalize_answer(question, "a,b;c") == "a,b;c"


@pytest.mark.parametrize(
    "question",
    [TIGHT_LIST_Q, "Comma-separated list, without spaces.", "Give a comma separated list (no whitespace)."],
)
def test_lists_use_bare_commas_when_the_question_forbids_spaces(question: str) -> None:
    assert normalize_answer(question, "a, b ;c") == "a,b,c"


@pytest.mark.parametrize("question", [LIST_Q, PLAIN])
def test_python_lists_are_joined_with_comma_and_space(question: str) -> None:
    assert normalize_answer(question, ["a", "b", 3]) == "a, b, 3"


def test_comma_lists_do_not_translate_sort_or_change_case() -> None:
    assert normalize_answer(LIST_Q, "zeta,Alpha,beta") == "zeta, Alpha, beta"


@pytest.mark.parametrize(("question", "raw", "expected"), INVENTED)
def test_invented_questions_end_to_end(question: str, raw: object, expected: str) -> None:
    assert normalize_answer(question, raw) == expected


def test_an_empty_question_still_gets_the_default_clean_up() -> None:
    assert normalize_answer("", "FINAL ANSWER: 1,234.") == "1234"


def test_normalising_twice_changes_nothing() -> None:
    corpus = [
        (PLAIN, raw)
        for raw in UNTOUCHED
        + KEPT_QUOTES
        + KEPT_PERIODS
        + NOT_THOUSANDS
        + [r for r, _ in LABELS + WRAPPERS + WHITESPACE + PERIODS + THOUSANDS]
    ]
    corpus += [(LIST_Q, raw) for raw, _ in LIST_CASES] + [(q, r) for q, r, _ in INVENTED]
    corpus += [(TIGHT_LIST_Q, "a ; b,c"), (LIST_Q, 'red, "blue."'), (LIST_Q, '"red", blue.')]

    for question, raw in corpus:
        once = normalize_answer(question, raw)
        assert normalize_answer(question, once) == once, (question, raw, once)
