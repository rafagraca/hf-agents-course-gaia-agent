"""Client for the course scoring API.

Fetches the question list (cached on disk), downloads question attachments and
submits answers. A submission is a dry run unless explicitly enabled.

Everything fetched here is benchmark material, so it is only ever written below
the DATA folder (``Settings.cache_dir``), never inside the repository. Calls have
timeouts and a bounded number of retries, downloads are size-limited, and file
names received from the API are treated as untrusted input (see ``_attachments``).
The checks on submitted answers live in ``_submission`` and are re-exported here.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from huggingface_hub import hf_hub_download

from gaia_agent._attachments import (
    UnsafeFileNameError,
    cached_copy,
    destination,
    sanitize_file_name,
    write_atomic,
)
from gaia_agent._submission import InvalidSubmissionError, build_payload, identity_problems, validate_answers
from gaia_agent.config import REPO_ROOT, ConfigError, Settings, get_secret

__all__ = [
    "ApiError",
    "InvalidSubmissionError",
    "Question",
    "ScoringClient",
    "UnsafeFileNameError",
    "sanitize_file_name",
    "validate_answers",
]

logger = logging.getLogger(__name__)

QUESTIONS_CACHE_NAME = "questions.json"
REQUEST_TIMEOUT_SECONDS = 30
SUBMIT_TIMEOUT_SECONDS = 180  # scoring a submission can take a while
FILE_TIMEOUT_SECONDS = 120  # per socket read, as in ``requests``
DOWNLOAD_DEADLINE_SECONDS = 180.0  # for a whole download, however slowly the server drips
MAX_FILE_BYTES = 20 * 1024 * 1024
CHUNK_BYTES = 64 * 1024
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0
MAX_DETAIL_CHARS = 300

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")
_TRANSIENT_ERRORS = (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError)


class ApiError(RuntimeError):
    """The scoring API (or the dataset fallback) could not deliver what was asked for."""


class _TransientError(Exception):
    """A failure worth retrying: a timeout, a connection problem or an HTTP 5xx."""


@dataclass(frozen=True)
class Question:
    """One exam question as served by the scoring API (``file_name`` is empty without an attachment)."""

    task_id: str
    question: str
    level: str
    file_name: str


# ----------------------------------------------------------------- reply handling


def _redact(text: str, secret: str) -> str:
    """Hide ``secret`` in ``text`` so that a token can never reach a log or an error message."""
    return text.replace(secret, "***")


def _error_detail(response: requests.Response) -> str:
    """Short, printable explanation taken from an error reply (FastAPI puts it under ``detail``)."""
    try:
        body = response.json()
    except ValueError:
        text = response.text
    else:
        text = str(body.get("detail", "")) if isinstance(body, dict) else ""
    text = _CONTROL_CHARS.sub(" ", text).strip()[:MAX_DETAIL_CHARS]
    return f": {text}" if text else ""


def _expect_json(response: requests.Response, what: str) -> Any:
    """The JSON body of a 200 reply; anything else is an :class:`ApiError`."""
    if response.status_code != 200:
        raise ApiError(f"{what} returned HTTP {response.status_code}{_error_detail(response)}")
    try:
        return response.json()
    except ValueError as exc:
        raise ApiError(f"{what} did not return valid JSON") from exc


def _read_limited(
    response: requests.Response,
    *,
    deadline_s: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> bytes:
    """Read the body of ``response``, refusing anything over ``MAX_FILE_BYTES`` or slower than the deadline in all.

    The deadline is ``deadline_s`` seconds (``DOWNLOAD_DEADLINE_SECONDS`` by default). The timeout of
    ``requests`` applies to each read of the socket, so a server that sends a byte every few seconds would
    otherwise hold the download for as long as it likes.
    """
    deadline_s = DOWNLOAD_DEADLINE_SECONDS if deadline_s is None else deadline_s
    deadline = clock() + deadline_s
    try:
        declared = int(response.headers.get("Content-Length", ""))
    except ValueError:
        declared = -1  # absent or malformed: the streaming count below still protects us
    if declared > MAX_FILE_BYTES:
        raise ApiError(f"file is {declared} bytes, over the {MAX_FILE_BYTES} byte limit")
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
        total += len(chunk)
        if total > MAX_FILE_BYTES:
            raise ApiError(f"file is larger than the {MAX_FILE_BYTES} byte limit")
        if clock() > deadline:
            raise ApiError(f"the download took too long (more than {deadline_s:g} s)")
        chunks.append(chunk)
    return b"".join(chunks)


def _accept_dataset_file(path: Path, root: Path) -> Path:
    """Check a file written by ``hf_hub_download``: inside ``root`` and within the size limit."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ApiError("dataset download landed outside the files folder")
    size = resolved.stat().st_size
    if size > MAX_FILE_BYTES:
        resolved.unlink()
        raise ApiError(f"dataset file is {size} bytes, over the {MAX_FILE_BYTES} byte limit")
    return resolved


# ------------------------------------------------------------------------ questions


def _parse_question(item: object, index: int) -> Question:
    """One entry of the question list; raise ``ValueError`` when it is malformed."""
    if not isinstance(item, Mapping):
        raise ValueError(f"question #{index} is not a JSON object")
    task_id, text = item.get("task_id"), item.get("question")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(f"question #{index} has no task_id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"question #{index} has no question text")
    file_name = item.get("file_name") or ""
    if not isinstance(file_name, str):
        raise ValueError(f"question #{index} has a file_name that is not text")
    level = item.get("Level", item.get("level"))  # the API spells it "Level"
    return Question(task_id=task_id, question=text, level="" if level is None else str(level), file_name=file_name)


def _parse_questions(payload: object) -> list[Question]:
    """The question list served by ``GET /questions``; raise ``ValueError`` when it is unusable."""
    if not isinstance(payload, list) or not payload:
        raise ValueError("expected a non-empty JSON list")
    return [_parse_question(item, index) for index, item in enumerate(payload)]


def _read_cached_questions(path: Path) -> list[Question] | None:
    """The cached questions, or None when there is no usable cache."""
    try:
        return _parse_questions(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring the unusable question cache %s (%s); downloading it again.", path.name, exc)
        return None


def _save_questions_cache(path: Path, payload: object) -> None:
    """Cache the API reply as served. The cache is an optimisation, so a failure is only logged."""
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        write_atomic(path, text.encode("utf-8"))
    except OSError as exc:
        logger.warning("Could not cache the questions in %s (%s).", path, exc.strerror or type(exc).__name__)


# ----------------------------------------------------------------------- the client


class ScoringClient:
    """Thin client for the scoring API; inject ``session`` to fake the network in tests.

    Failures raise :class:`ApiError`, except :meth:`download_file`, which returns None
    and logs the reason, because a missing attachment must not stop a whole run.
    ``sleep`` is the wait between retries (injected so tests need not sleep).
    """

    def __init__(
        self,
        settings: Settings,
        session: requests.Session | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        data_dir = settings.data_dir.resolve()
        if data_dir.is_relative_to(REPO_ROOT):
            raise ConfigError(
                f"the data folder must be outside the repository (benchmark data must never be committed): {data_dir}"
            )
        self._settings = settings
        self._session = session if session is not None else requests.Session()
        self._sleep = sleep

    def get_questions(self, refresh: bool = False) -> list[Question]:
        """Return the questions, cached in ``<data>/cache/questions.json`` (``refresh`` forces a new download)."""
        cache = self._settings.cache_dir / QUESTIONS_CACHE_NAME
        if not refresh:
            cached = _read_cached_questions(cache)
            if cached is not None:
                return cached
        url = self._url("questions")
        payload = self._request_json("GET", url, timeout=REQUEST_TIMEOUT_SECONDS)
        try:
            questions = _parse_questions(payload)
        except ValueError as exc:
            raise ApiError(f"unexpected question list from {url}: {exc}") from exc
        _save_questions_cache(cache, payload)
        logger.info("Fetched %d questions from %s", len(questions), url)
        return questions

    def download_file(self, q: Question) -> Path | None:
        """Download the attachment of ``q`` into ``<data>/cache/files``.

        Tries ``GET {api}/files/{task_id}`` first. When that does not answer 200 and an
        ``HF_TOKEN`` is available, falls back to ``huggingface_hub.hf_hub_download`` from the
        dataset repo. The file name is sanitised against path traversal, downloads are limited
        to 20 MB, and a file fetched before is reused. Returns None, and logs the reason, when
        the file cannot be obtained.
        """
        if not q.file_name:
            return None
        try:
            name = sanitize_file_name(q.file_name)
            root = self._files_root()
            cached = cached_copy(root, self._settings.hf_split_dir, name)
        except (UnsafeFileNameError, OSError) as exc:
            logger.warning("Cannot fetch the attachment of task %r: %s", q.task_id, exc)
            return None
        if cached is not None:
            return cached
        reasons: list[str] = []
        for fetch in (
            lambda: self._download_from_api(q.task_id, name, root),
            lambda: self._download_from_dataset(name, root),
        ):
            try:
                return fetch()
            except (ApiError, UnsafeFileNameError, OSError) as exc:
                reasons.append(str(exc))
        logger.warning("No attachment for task %r (%s): %s", q.task_id, name, "; ".join(reasons))
        return None

    def submit(
        self, username: str, agent_code: str, answers: list[dict[str, Any]], dry_run: bool = True
    ) -> dict[str, Any]:
        """Validate ``answers`` and, only when ``dry_run`` is False, POST them to ``{api}/submit``.

        Validation uses the cached question list (downloaded once if missing) and raises
        :class:`InvalidSubmissionError` listing every problem. A dry run returns
        ``{"dry_run": True, "url": ..., "payload": ...}`` with the exact payload it would send;
        a real submission returns the scoring API's JSON reply. Only ``task_id`` and
        ``submitted_answer`` of each answer are ever sent. Nothing is posted unless
        ``dry_run`` is exactly ``False``. A retried POST repeats the same payload, so the API
        just scores the same answers again.
        """
        problems = [*identity_problems(username, agent_code), *validate_answers(self.get_questions(), answers)]
        if problems:
            raise InvalidSubmissionError(problems)
        payload = build_payload(username, agent_code, answers)
        url = self._url("submit")
        if dry_run is not False:  # a sloppy None or 0 must never turn into a real submission
            logger.info("Dry run: %d answers are valid, nothing was sent.", len(answers))
            return {"dry_run": True, "url": url, "payload": payload}
        reply = self._request_json("POST", url, timeout=SUBMIT_TIMEOUT_SECONDS, body=payload)
        if not isinstance(reply, dict):
            raise ApiError(f"POST {url} did not return a JSON object")
        return reply

    # -- attachment sources -------------------------------------------------------

    def _files_root(self) -> Path:
        root = self._settings.files_dir
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    def _download_from_api(self, task_id: str, name: str, root: Path) -> Path:
        target = destination(root, name)  # refuse a bad destination before any request
        url = self._url("files", task_id)
        data = self._with_retries(f"GET {url}", lambda: self._fetch_file(url))
        write_atomic(target, data)
        return target

    def _download_from_dataset(self, name: str, root: Path) -> Path:
        token = get_secret("HF_TOKEN")
        if token is None:
            raise ApiError("dataset fallback skipped: HF_TOKEN is not set")
        repo = self._settings.hf_dataset
        try:
            downloaded = hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=f"{self._settings.hf_split_dir}/{name}",
                token=token,
                local_dir=root,
            )
        except Exception as exc:  # noqa: BLE001 - best effort: whatever went wrong is reported, never raised raw
            # ``from None`` keeps the original exception, whose text could echo the token, out of tracebacks.
            raise ApiError(f"dataset {repo}: {type(exc).__name__}: {_redact(str(exc), token)}") from None
        return _accept_dataset_file(Path(downloaded), root)

    # -- HTTP plumbing ------------------------------------------------------------

    def _url(self, *segments: str) -> str:
        base = self._settings.api_url.rstrip("/")
        return "/".join([base, *(quote(segment, safe="") for segment in segments)])

    def _send(
        self, method: str, url: str, *, timeout: float, stream: bool = False, body: object = None
    ) -> requests.Response:
        """One request. Timeouts, connection problems and HTTP 5xx raise ``_TransientError``."""
        try:
            # A POST carries the username and the answers: it is never sent on to wherever a redirect points.
            response = self._session.request(
                method, url, timeout=timeout, stream=stream, json=body, allow_redirects=method != "POST"
            )
        except _TRANSIENT_ERRORS as exc:
            raise _TransientError(type(exc).__name__) from exc
        except (requests.RequestException, ValueError) as exc:
            # ValueError: ``requests`` does not wrap a malformed redirect address (e.g. "Invalid IPv6 URL").
            raise ApiError(f"{method} {url} failed: {type(exc).__name__}") from exc
        if response.status_code >= 500:
            response.close()
            raise _TransientError(f"HTTP {response.status_code}")
        return response

    def _with_retries[T](self, what: str, attempt: Callable[[], T]) -> T:
        """Run ``attempt`` up to ``MAX_ATTEMPTS`` times, waiting 1 s, 2 s, ... after transient failures."""
        reason = ""
        for number in range(1, MAX_ATTEMPTS + 1):
            try:
                return attempt()
            except _TransientError as exc:
                reason = str(exc)
            if number < MAX_ATTEMPTS:
                delay = RETRY_BASE_DELAY_SECONDS * 2 ** (number - 1)
                logger.warning(
                    "%s failed (%s), attempt %d of %d; retrying in %.0f s", what, reason, number, MAX_ATTEMPTS, delay
                )
                self._sleep(delay)
        raise ApiError(f"{what} failed after {MAX_ATTEMPTS} attempts ({reason})")

    def _request_json(self, method: str, url: str, *, timeout: float, body: object = None) -> Any:
        def attempt() -> Any:
            return _expect_json(self._send(method, url, timeout=timeout, body=body), f"{method} {url}")

        return self._with_retries(f"{method} {url}", attempt)

    def _fetch_file(self, url: str) -> bytes:
        """One attempt at downloading a file; a body that breaks midway counts as transient."""
        response = self._send("GET", url, timeout=FILE_TIMEOUT_SECONDS, stream=True)
        try:
            if response.status_code != 200:
                raise ApiError(f"GET {url} returned HTTP {response.status_code}{_error_detail(response)}")
            return _read_limited(response)
        except _TRANSIENT_ERRORS as exc:
            raise _TransientError(type(exc).__name__) from exc
        finally:
            response.close()
