"""Normalise raw agent output into the exact-match answer string the scorer expects.

The scorer compares the submitted string with the reference answer, so the only
useful job here is removing the formatting noise an agent tends to add around a
correct answer: an "answer" label, wrapping quotes, a stray full stop, thousands
separators and sloppy list separators.

The clean-up is deliberately conservative. When in doubt the text is left alone,
and nothing is ever translated, re-cased, sorted or rounded. The question is only
read for explicit format requests (a comma-separated list, thousands separators,
decimals); it never selects an answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["normalize_answer"]

MAX_PASSES = 5
"""Upper bound on clean-up passes; the text normally settles after one or two."""

MAX_INTEGRAL_FLOAT = 1e15
"""Floats at or above this magnitude keep their ``str`` form (no int conversion)."""

# --- What the question says about the expected format -------------------------------------------

_LIST_REQUEST = re.compile(
    r"comma[\s\-\u2013]*(?:separated|delimited)"
    r"|(?:separated|delimited)\s+(?:by|with|using)\s+(?:an?\s+|the\s+)?commas?\b"
    r"|\bseparate\w*\s+(?:\w+\s+){0,4}(?:by|with|using)\s+(?:an?\s+)?commas?\b",
    re.IGNORECASE,
)
_NO_SPACES = re.compile(r"\b(?:no|without)\s+(?:any\s+)?(?:white\s*space|spaces?)\b", re.IGNORECASE)
_THOUSANDS_FORMAT = re.compile(
    r"thousands?[\s\-]*(?:separators?|delimiters?)"
    r"|digit[\s\-]*group"
    r"|\bthousands?\b[^.?!]{0,40}\bcommas?\b"
    r"|\bcommas?\b[^.?!]{0,40}\bthousands?\b",
    re.IGNORECASE,
)
_WITH_COMMAS = re.compile(r"\b(?:with|using|include|including)\s+commas?\b", re.IGNORECASE)
_SPACE_SEPARATED = re.compile(
    r"space[\s\-]*(?:separated|delimited)|(?:separated|delimited)\s+(?:by|with)\s+(?:an?\s+)?(?:spaces?|whitespace)",
    re.IGNORECASE,
)
_DECIMALS = re.compile(r"decimal|significant|precision|\b(?:tenths?|hundredths?|thousandths?)\b", re.IGNORECASE)
_ROUND = re.compile(r"\bround", re.IGNORECASE)
# "Round to the nearest integer" asks for no decimals at all.
_ROUND_TO_WHOLE = re.compile(
    r"nearest\s+(?:integer|whole(?:\s+number)?|ten|hundred|thousand|million|dollar)s?\b", re.IGNORECASE
)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_COUNT = r"(\d+|" + "|".join(_NUMBER_WORDS) + r")"
# "two decimal places" and "to 3 decimals" ask for a format; "the two decimals" in "which of the two decimals
# is bigger?" does not.
_PLACES = re.compile(
    rf"\b{_COUNT}\s+decimal\s+(?:places?|digits?)\b|\b(?:to|with|using|at)\s+{_COUNT}\s+decimals?\b", re.IGNORECASE
)
_NEAREST_PLACE = re.compile(r"nearest\s+(tenth|hundredth|thousandth)s?\b", re.IGNORECASE)
_PLACES_OF_NEAREST = {"tenth": 1, "hundredth": 2, "thousandth": 3}
_PLAIN_NUMBER = re.compile(r"([+-]?\d+)(?:\.(\d*))?")


@dataclass(frozen=True)
class _Hints:
    """The format requests found in the question."""

    comma_list: bool = False
    tight_list: bool = False  # the list must use bare commas, without a space after them
    keep_thousands: bool = False
    space_separated: bool = False
    decimals: bool = False
    places: int | None = None  # the number of decimal places asked for, when the question gives one


def _wants_decimals(text: str) -> bool:
    """Whether the question talks about decimals; rounding to a whole number does not count."""
    if _DECIMALS.search(text) is not None:
        return True
    return _ROUND.search(text) is not None and _ROUND_TO_WHOLE.search(text) is None


def _decimal_places(text: str) -> int | None:
    """The number of decimal places the question asks for ("two decimal places", "the nearest hundredth")."""
    found = _PLACES.search(text)
    if found is not None:
        word = (found.group(1) or found.group(2)).lower()
        return _NUMBER_WORDS[word] if word in _NUMBER_WORDS else int(word)
    nearest = _NEAREST_PLACE.search(text)
    return _PLACES_OF_NEAREST[nearest.group(1).lower()] if nearest else None


def _read_hints(question: str) -> _Hints:
    text = question if isinstance(question, str) else ""
    thousands_format = _THOUSANDS_FORMAT.search(text) is not None
    # A request about thousands separators beats a list request: a list cannot be re-spaced
    # safely when its numbers contain commas.
    comma_list = _LIST_REQUEST.search(text) is not None and not thousands_format
    return _Hints(
        comma_list=comma_list,
        tight_list=comma_list and _NO_SPACES.search(text) is not None,
        keep_thousands=thousands_format or _WITH_COMMAS.search(text) is not None,
        space_separated=_SPACE_SEPARATED.search(text) is not None,
        decimals=_wants_decimals(text),
        places=None if comma_list else _decimal_places(text),
    )


# --- Raw object -> text -------------------------------------------------------------------------


def _stringify_float(value: float, hints: _Hints) -> str:
    """``str(value)``, except that an integral float is written as an integer.

    ``5.0`` is an artefact of Python's default formatting, not a formatting choice of
    the agent. It stays ``5.0`` when the question talks about decimals or rounding.
    """
    if value.is_integer() and abs(value) < MAX_INTEGRAL_FLOAT and not hints.decimals:
        return str(int(value))
    return str(value)


def _stringify(raw: object, hints: _Hints) -> str:
    """Convert any object to text; sequences become a ``", "``-joined list."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, bytes | bytearray):
        return bytes(raw).decode("utf-8", errors="replace")
    if isinstance(raw, bool):
        return str(raw)
    if isinstance(raw, float):
        return _stringify_float(raw, hints)
    if isinstance(raw, list | tuple):
        return ", ".join(_stringify(item, hints) for item in raw)
    if isinstance(raw, set | frozenset):
        return ", ".join(sorted(_stringify(item, hints) for item in raw))
    return str(raw)


# --- Text clean-up ------------------------------------------------------------------------------

# No-break space, the U+2000..U+200A block of typographic spaces, narrow no-break space, medium mathematical space.
_SPACE_LIKE = {code: " " for code in (0x00A0, *range(0x2000, 0x200B), 0x202F, 0x205F)}
# Zero-width space, word joiner and byte-order mark: invisible, never part of an expected answer.
_INVISIBLE = dict.fromkeys((0x200B, 0x2060, 0xFEFF))
_SPACE_RUN = re.compile(r"[ \t]{2,}")


def _clean_whitespace(text: str) -> str:
    """Normalise line breaks and exotic spaces, drop zero-width characters and trim."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate({**_SPACE_LIKE, **_INVISIBLE})
    return _SPACE_RUN.sub(" ", text).strip()


_FINAL_MARKER = re.compile(
    r"final\s+answer\b[*_]{0,2}(?:\s+is\b\s*[:\uff1a]?|\s*[:\uff1a])[*_]{0,2}",
    re.IGNORECASE,
)  # "FINAL ANSWER:" or "final answer is", anywhere in the text
_FINAL_LEADING = re.compile(
    r"^[*_]{0,2}(?:the\s+)?final\s+answer\b[*_]{0,2}(?:\s+is\b)?\s*[:\uff1a\-\u2013\u2014]?[*_]{0,2}\s*",
    re.IGNORECASE,
)  # a leading "Final answer", with or without punctuation
_ANSWER_LABEL = re.compile(r"^[*_]{0,2}answer\b[*_]{0,2}\s*[:\uff1a][*_]{0,2}\s*", re.IGNORECASE)


def _strip_labels(text: str) -> str:
    """Drop a "FINAL ANSWER:" style label, keeping what follows its last occurrence.

    The submission check rejects any "final answer" text, so the label forms agents
    actually produce are all removed. A lone "Answer:" is only a label at the very start.
    """
    last_end = None
    for match in _FINAL_MARKER.finditer(text):
        last_end = match.end()
    if last_end is not None:
        return text[last_end:].lstrip()
    for label in (_FINAL_LEADING, _ANSWER_LABEL):
        stripped = label.sub("", text, count=1)
        if stripped != text:
            return stripped
    return text


_QUOTE_PAIRS = {'"': '"', "'": "'", "`": "`", "\u201c": "\u201d", "\u2018": "\u2019", "\u00ab": "\u00bb"}
_FENCE = "```"
_FENCE_LANGUAGES = frozenset(
    {"", "text", "txt", "plain", "plaintext", "markdown", "md", "python", "py", "json", "csv", "bash", "sh", "latex"}
)


def _strip_fence(text: str) -> str:
    body = text[len(_FENCE) : -len(_FENCE)]
    first_line, newline, rest = body.partition("\n")
    if newline and first_line.strip().lower() in _FENCE_LANGUAGES:
        body = rest
    return body.strip()


def _unwrap(text: str) -> str:
    """Remove one layer of code fence, bold markers or matching quotes around the whole text.

    Quotes are only removed when they are a clean pair: the same quote character
    appearing inside (``"a" and "b"``, ``'It's fine'``) leaves the text untouched.
    """
    if len(text) >= 2 * len(_FENCE) and text.startswith(_FENCE) and text.endswith(_FENCE):
        return _strip_fence(text)
    if len(text) >= 5 and text.startswith("**") and text.endswith("**") and "**" not in text[2:-2]:
        return text[2:-2].strip()
    close = _QUOTE_PAIRS.get(text[:1])
    if close is not None and len(text) >= 2 and text.endswith(close):
        inner = text[1:-1]
        if text[0] not in inner and close not in inner:
            return inner.strip()
    return text


_DOTTED_ABBREVIATION = re.compile(r"(?:[^\W\d_]{1,4}\.)+[^\W\d_]{1,4}")  # U.S., Ph.D., a.m.
_ABBREVIATIONS = frozenset(
    {
        "jr",
        "sr",
        "inc",
        "ltd",
        "co",
        "corp",
        "st",
        "mt",
        "ft",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "rev",
        "gen",
        "col",
        "sgt",
        "capt",
        "lt",
        "gov",
        "sen",
        "rep",
        "hon",
        "vs",
        "etc",
        "vol",
        "fig",
    }
)  # "no" is deliberately absent: "No." is a plain answer.
_EDGE_PUNCTUATION = "\"'()[]{}\u201c\u201d\u2018\u2019\u00ab\u00bb"


def _ends_with_abbreviation(body: str) -> bool:
    """Whether ``body`` ends with an abbreviation or an initial that owns its final period."""
    if body.lower().endswith("et al"):
        return True
    last = body.split()[-1].strip(_EDGE_PUNCTUATION)
    if _DOTTED_ABBREVIATION.fullmatch(last):
        return True
    return (len(last) == 1 and last.isalpha()) or last.lower() in _ABBREVIATIONS


def _strip_final_period(text: str) -> str:
    """Drop one loose full stop at the end, but not an ellipsis, an abbreviation or an initial."""
    if not text.endswith(".") or text.endswith(".."):
        return text
    body = text[:-1].rstrip()
    if not any(ch.isalnum() for ch in body) or _ends_with_abbreviation(body):
        return text
    return body


def _clean(text: str) -> str:
    """One pass of the format-independent clean-up."""
    text = _unwrap(_clean_whitespace(text))
    return _strip_final_period(_unwrap(_strip_labels(text)))


# --- Numbers and lists --------------------------------------------------------------------------

_COMMA_NUMBER = re.compile(r"[+-]?[1-9][0-9]{0,2}(?:,[0-9]{3})+(?:\.[0-9]+)?")  # ASCII digits only
_SPACE_NUMBER = re.compile(r"[+-]?[1-9][0-9]{0,2}(?: [0-9]{3})+(?:\.[0-9]+)?")


def _merge_space_groups(item: str, hints: _Hints) -> str:
    """``1 234`` -> ``1234``, unless the question wants space-separated values or separators."""
    if hints.keep_thousands or hints.space_separated or not _SPACE_NUMBER.fullmatch(item):
        return item
    return item.replace(" ", "")


def _merge_thousands(text: str, hints: _Hints) -> str:
    """Remove thousands separators when the whole answer is a single grouped number."""
    if hints.keep_thousands:
        return text
    if _COMMA_NUMBER.fullmatch(text):
        return text.replace(",", "")
    return _merge_space_groups(text, hints)


_LIST_SPLIT = re.compile(r"\s*[,;\n]\s*")


def _normalize_list(text: str, hints: _Hints) -> str:
    """Re-join a list asked for as "comma-separated" with uniform separators.

    Every comma is a separator here (``1,234`` is the list ``1, 234``): the question
    asked for a list, and splitting is the only reading that never merges items.
    """
    if text.startswith("[") and text.endswith("]") and not re.search(r"[\[\]]", text[1:-1]):
        text = text[1:-1]
    items = (_merge_space_groups(_unwrap(part.strip()), hints) for part in _LIST_SPLIT.split(text))
    return ("," if hints.tight_list else ", ").join(item for item in items if item)


def _normalize_text(text: str, hints: _Hints) -> str:
    text = _clean(text)
    if hints.comma_list:
        return _normalize_list(text, hints)
    return _merge_thousands(text, hints)


def _fill_decimals(text: str, places: int | None) -> str:
    """``text`` with zeros added to its decimal part up to ``places`` digits, when it is a plain number.

    ``12.6`` asked "to two decimal places" is ``12.60``. Nothing is ever rounded or cut: a number that has
    as many decimals as asked, or more, is left as it is, and so is any text that is not a plain number.
    """
    found = _PLAIN_NUMBER.fullmatch(text) if places else None
    if found is None:
        return text
    whole, fraction = found.group(1), found.group(2) or ""
    return text if len(fraction) >= places else f"{whole}.{fraction.ljust(places, '0')}"


def normalize_answer(question: str, raw: object) -> str:
    """Return the answer string to submit for the raw agent output ``raw``.

    ``question`` gives context about the expected format. Pure function that
    always returns a ``str``. It is conservative: what it changes is limited to

    * conversion to text (``None`` -> ``""``, sequences -> ``"a, b"``, integral
      floats -> ``"5"`` unless the question talks about decimals; rounding to a
      whole number does not count as talking about decimals);
    * removing a ``FINAL ANSWER:`` / ``Answer:`` label, stray whitespace, one layer of
      wrapping quotes, backticks, code fence or bold markers, and one loose final
      period (kept after abbreviations, initials, decimals and ellipses);
    * ``1,234`` / ``1 234`` -> ``1234`` for a lone number, unless the question asks
      for thousands separators;
    * zeros added to a lone number up to the number of decimal places the question
      asks for (``12.6`` -> ``12.60``; more decimals than asked are left as they are);
    * for a question asking for a comma-separated list: uniform ``", "`` separators
      (bare ``","`` when the question forbids spaces).

    The text is never translated, re-cased, sorted or rounded.
    """
    hints = _read_hints(question)
    text = _stringify(raw, hints)
    for _ in range(MAX_PASSES):
        updated = _normalize_text(text, hints)
        if updated == text:
            break
        text = updated
    return _fill_decimals(text, hints.places)
