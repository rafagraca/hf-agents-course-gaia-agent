"""Sources the web tools refuse to read: places that publish the answers of the benchmark.

The exam is meant to be answered by finding the answers, not by copying them. Other students' answers sit in
public files (a ``answers.jsonl`` in a Space, a GitHub repository of solutions, the benchmark dataset itself, a
forum thread), and a web search made with the words of a question can lead straight to them. A reviewer reading
the traces must be able to see that the agent never opened such a place, so:

* :func:`blocked_source` judges an address (the first one, every redirect hop and every search result): the
  benchmark dataset, Spaces and repositories named after the benchmark or the course exercise, and the forum
  threads about them. It looks at the host, the path and the query only, and no address of an ordinary site is affected,
  even one that talks about Gaia the space telescope or about answers in general;
* :func:`looks_like_answer_key` judges the text of a page that was fetched anyway (a mirror under another
  name): task ids next to final answers are the fingerprint of the scoring files.

This is a rule about *where* to read, not about any question: nothing here knows a single answer or task.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

__all__ = ["BLOCKED_MESSAGE", "blocked_source", "looks_like_answer_key"]

BLOCKED_MESSAGE = "blocked source: this place publishes the answers of the benchmark; find the answer elsewhere"

_BENCHMARK_WORDS = re.compile(r"gaia|unit-?4|final[-_ ]?assignment|agents?[-_]course|answers?(?![a-z])", re.IGNORECASE)
_REPOSITORY_WORDS = re.compile(r"gaia|unit-?4|final[-_ ]?assignment|agents?[-_]course", re.IGNORECASE)
_REPOSITORY_HOSTS = frozenset(
    {
        "github.com",
        "gist.github.com",
        "raw.githubusercontent.com",
        "gitlab.com",
        "bitbucket.org",
        "kaggle.com",
        "www.kaggle.com",
    }
)
# The Hub, its short name and the usual mirror: the same repositories, the same API.
_HUGGING_FACE_HOSTS = frozenset(
    {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co", "hf-mirror.com", "www.hf-mirror.com"}
)
_ROWS_SERVER_HOST = "datasets-server.huggingface.co"  # rows of any dataset, named in the query
_HUGGING_FACE_REPOSITORY_KINDS = frozenset({"spaces", "datasets"})  # their names are judged by the wider word list
_HUGGING_FACE_READING_SECTIONS = frozenset({"learn", "docs", "blog", "papers"})  # the course lessons, never blocked
_FORUM_HOST = "discuss.huggingface.co"
_SPACE_HOST_SUFFIX = ".hf.space"

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_UUIDS = re.compile(_UUID)
_KEYED_UUID = re.compile(
    rf"(?:task[_\s-]?id|submitted_answer|final[_\s-]?answer)[\"']?\s*[:=|,]?\s*[\"']?\s*{_UUID}", re.IGNORECASE
)
_ANSWER_WORDS = re.compile(r"final[_\s-]?answer|submitted_answer", re.IGNORECASE)
_TASK_WORDS = re.compile(r"task[_\s-]?id", re.IGNORECASE)


def _host_path_query(url: str) -> tuple[str, str, str] | None:
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower().rstrip(".")
    except ValueError:
        return None
    return (host, parts.path.lower(), parts.query.lower()) if host else None


def blocked_source(url: str) -> str | None:
    """Why ``url`` may not be read (a short message for the agent), or None when it may.

    Blocked: on the Hub (and its mirror and rows server, the ``/api/`` paths and queries included) any Space,
    dataset or repository of any kind named after the benchmark or the course exercise, and Spaces and datasets
    named after answers; Spaces served from ``*.hf.space`` with such a name; forum threads about them; and
    repositories on the usual code and data hosts that carry the name of the benchmark or the exercise.
    """
    located = _host_path_query(url)
    if located is None:
        return None
    if _is_benchmark_place(*located):
        return BLOCKED_MESSAGE
    return None


def _is_hugging_face_benchmark_place(path: str, query: str) -> bool:
    segments = [segment for segment in path.split("/") if segment]
    if segments[:1] == ["api"]:
        segments = segments[1:]
    if segments and segments[0] in _HUGGING_FACE_READING_SECTIONS:
        return False
    words = _BENCHMARK_WORDS if segments and segments[0] in _HUGGING_FACE_REPOSITORY_KINDS else _REPOSITORY_WORDS
    return words.search(f"{path}?{query}") is not None


def _is_benchmark_place(host: str, path: str, query: str) -> bool:
    if host in _HUGGING_FACE_HOSTS:
        return _is_hugging_face_benchmark_place(path, query)
    if host == _ROWS_SERVER_HOST:
        return _BENCHMARK_WORDS.search(f"{path}?{query}") is not None
    if host == _FORUM_HOST:
        return _BENCHMARK_WORDS.search(path) is not None
    if host.endswith(_SPACE_HOST_SUFFIX):
        return _BENCHMARK_WORDS.search(host.removesuffix(_SPACE_HOST_SUFFIX)) is not None
    if host in _REPOSITORY_HOSTS:
        return _REPOSITORY_WORDS.search(path) is not None
    return False


def looks_like_answer_key(text: str) -> bool:
    """True when ``text`` reads like a file of task ids and answers of the benchmark's scoring.

    Two task ids that are each introduced by ``task_id`` or ``final answer`` (the shape of a JSON Lines
    file), or one task id in a text that also has the words ``task_id`` and ``final answer`` (a table, a
    metadata row). A single ``task_id`` in a tutorial, or an article about final answers, is no answer key.
    """
    if not text:
        return False
    if len(_KEYED_UUID.findall(text)) >= 2:
        return True
    return (
        _UUIDS.search(text) is not None
        and _ANSWER_WORDS.search(text) is not None
        and _TASK_WORDS.search(text) is not None
    )
