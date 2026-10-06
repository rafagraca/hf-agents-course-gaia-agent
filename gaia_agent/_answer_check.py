"""The check the agent runs on every ``final_answer`` before it accepts it.

smolagents 1.26 runs ``check_function(final_answer, self.memory, agent=self)`` inside an ``assert``; when it is
falsy the step ends with the error "Check <function name> failed with error: " and the model tries again. The
text after "error:" is empty for a plain ``False``, so the name of the function is all the model learns: it
says what is wanted.

The scorer compares the answer to a short reference, letter by letter, so an answer is refused when it is
empty, carries the "FINAL ANSWER" label, spans several lines, is a refusal or is a sentence (more than
``MAX_WORDS_WITHOUT_COMMA`` words and no comma, the way a list would have). The rules are general: none looks
at what a question asks.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["MAX_WORDS_WITHOUT_COMMA", "check", "check_final_answer", "final_answer_must_be_a_short_bare_answer"]

MAX_WORDS_WITHOUT_COMMA = 12

_LABEL = re.compile(r"final\s+answer", re.IGNORECASE)
_LINE_BREAK = re.compile(r"[\r\n]")
_REFUSAL = re.compile(
    r"\b(?:"
    r"i\s+cannot|i\s+can\s*not|i\s+can't|"
    r"unable\s+to|"
    r"i\s+don't\s+know|i\s+do\s+not\s+know|"
    r"not\s+possible\s+to\s+determine|cannot\s+be\s+determined"
    r")\b",
    re.IGNORECASE,
)


def _as_text(final_answer: object) -> str:
    """The text of an answer as the scorer would see it, with typographic apostrophes made plain."""
    text = "" if final_answer is None else str(final_answer)
    return text.replace("’", "'").replace("‘", "'").strip()


def final_answer_must_be_a_short_bare_answer(final_answer: Any, memory: Any, agent: Any = None) -> bool:
    """True when ``final_answer`` looks like a short, bare answer; False makes the agent try again.

    Signature as smolagents 1.26 calls it: ``(final_answer, memory, agent=...)``; ``memory`` and ``agent`` are
    not used.
    """
    text = _as_text(final_answer)
    if not text or _LABEL.search(text) or _LINE_BREAK.search(text) or _REFUSAL.search(text):
        return False
    return "," in text or len(text.split()) <= MAX_WORDS_WITHOUT_COMMA


# The name of the function above is long on purpose: it is all the model reads of a refusal. These are the short ones.
check_final_answer = check = final_answer_must_be_a_short_bare_answer
