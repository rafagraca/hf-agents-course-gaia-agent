"""YouTube helpers: video ids, transcripts, and the transcript lines that matter for a query.

Transcripts come from the ``youtube-transcript-api`` package (v1.x:
``YouTubeTranscriptApi().list(video_id)``, then ``Transcript.fetch()``). Every
failure is turned into a short :class:`ToolError`, so the agent is told plainly
when a video has no captions instead of being tempted to invent its content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse
from xml.etree.ElementTree import ParseError

import requests
from youtube_transcript_api import (
    AgeRestricted,
    FetchedTranscript,
    InvalidVideoId,
    NoTranscriptFound,
    PoTokenRequired,
    RequestBlocked,
    Transcript,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeTranscriptApi,
    YouTubeTranscriptApiException,
)

from gaia_agent.tools._local_files import ToolError, truncate_text

YOUTUBE_TIMEOUT_S = 30
TRANSCRIPT_LIMIT = 6000
NO_MATCH_LIMIT = 2000
CONTEXT_LINES = 2
MAX_MATCHES = 8

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_SHORT_HOSTS = frozenset({"youtu.be", "www.youtu.be"})
_YOUTUBE_DOMAINS = ("youtube.com", "youtube-nocookie.com")
_ID_PATH_PREFIXES = frozenset({"embed", "shorts", "live", "v"})
_STOP_WORDS = frozenset(
    """
    a an and are as at be by for from how in is it of on or that the this to was what when where which
    who why with
    """.split()
)

# More specific exceptions first: IpBlocked is a RequestBlocked, for instance.
_FAILURES: tuple[tuple[type[Exception], str], ...] = (
    (TranscriptsDisabled, "subtitles are disabled for this video, so there is no transcript"),
    (NoTranscriptFound, "this video has no transcript"),
    (AgeRestricted, "this video is age-restricted, so its transcript cannot be retrieved"),
    (VideoUnavailable, "this video is unavailable (removed or private)"),
    (VideoUnplayable, "this video cannot be played (private, removed or blocked in this region)"),
    (InvalidVideoId, "the video id is invalid"),
    (RequestBlocked, "YouTube is blocking transcript requests from this network; try again later"),
    (PoTokenRequired, "YouTube refused to serve this transcript"),
)


# --------------------------------------------------------------------------- video ids


def canonical_url(video_id: str) -> str:
    """The plain watch URL of ``video_id``, without tracking parameters or time offsets."""
    return f"https://www.youtube.com/watch?v={video_id}"


def _is_youtube_host(host: str) -> bool:
    return any(host == domain or host.endswith(f".{domain}") for domain in _YOUTUBE_DOMAINS)


def _candidate_id(url: str) -> str | None:
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
    except ValueError:  # e.g. an unbalanced "[" makes urlparse reject the address
        return None
    host = (parsed.hostname or "").lower()
    segments = [segment for segment in parsed.path.split("/") if segment]
    if host in _SHORT_HOSTS:
        return segments[0] if segments else None
    if not _is_youtube_host(host) or not segments:
        return None
    if segments[0] == "watch":
        return next(iter(parse_qs(parsed.query).get("v", [])), None)
    if segments[0] in _ID_PATH_PREFIXES and len(segments) >= 2:
        return segments[1]
    return None


def parse_video_id(url: object) -> str | None:
    """Extract the 11-character video id from any common YouTube URL (or a bare id); None if there is none."""
    if not isinstance(url, str):
        return None
    text = url.strip().strip("\"'").strip()
    if _VIDEO_ID.fullmatch(text):
        return text
    candidate = _candidate_id(text)
    return candidate if candidate and _VIDEO_ID.fullmatch(candidate) else None


# --------------------------------------------------------------------------- fetching


def _pick_transcript(transcripts: list[Transcript]) -> Transcript:
    """English first, then any language; within a language, manual captions before automatic ones."""

    def rank(transcript: Transcript) -> tuple[int, int]:
        is_english = transcript.language_code.lower().split("-")[0] == "en"
        return (0 if is_english else 1, 1 if transcript.is_generated else 0)

    return min(transcripts, key=rank)


def _explain(exc: YouTubeTranscriptApiException) -> str:
    for kind, message in _FAILURES:
        if isinstance(exc, kind):
            return message
    return f"the transcript could not be retrieved ({type(exc).__name__})"


class _TimeoutSession(requests.Session):
    """A session that never waits forever: the transcript library sets no timeout of its own."""

    def request(self, *args: Any, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", YOUTUBE_TIMEOUT_S)
        return super().request(*args, **kwargs)


def fetch_transcript(video_id: str) -> FetchedTranscript:
    """Download the best available transcript of ``video_id``; raise :class:`ToolError` when there is none."""
    try:
        transcripts = list(YouTubeTranscriptApi(http_client=_TimeoutSession()).list(video_id))
        if not transcripts:
            raise ToolError("this video has no transcript")
        return _pick_transcript(transcripts).fetch()
    except YouTubeTranscriptApiException as exc:
        raise ToolError(_explain(exc)) from exc
    except requests.RequestException as exc:
        raise ToolError("could not reach YouTube") from exc
    except ParseError as exc:
        raise ToolError("YouTube returned an empty or unreadable transcript") from exc


# --------------------------------------------------------------------------- rendering


@dataclass(frozen=True)
class Line:
    """One caption line with its start time in seconds."""

    start: float
    text: str

    def render(self) -> str:
        minutes, seconds = divmod(int(self.start), 60)
        return f"[{minutes:02d}:{seconds:02d}] {self.text}"


def _lines_of(transcript: FetchedTranscript) -> list[Line]:
    cleaned = (Line(snippet.start, " ".join(snippet.text.split())) for snippet in transcript)
    return [line for line in cleaned if line.text]


def _header(transcript: FetchedTranscript, lines: list[Line]) -> str:
    kind = "auto-generated" if transcript.is_generated else "manual"
    language = transcript.language_code
    if language.lower().split("-")[0] != "en":
        language = f"{language}, not English"
    minutes, seconds = divmod(int(lines[-1].start), 60)
    return (
        f"YouTube transcript of {transcript.video_id} ({language}; {kind}): "
        f"{len(lines)} lines, {minutes:02d}:{seconds:02d} long"
    )


def _query_terms(query: str) -> list[str]:
    words = list(dict.fromkeys(re.findall(r"\w+", query.lower())))
    useful = [word for word in words if (len(word) > 2 or word.isdigit()) and word not in _STOP_WORDS]
    return useful or words


def _best_matches(lines: list[Line], terms: list[str]) -> list[int]:
    """Indexes of the lines that match most query terms (at most MAX_MATCHES), in transcript order."""
    scored = [(sum(term in line.text.lower() for term in terms), index) for index, line in enumerate(lines)]
    matching = [(score, index) for score, index in scored if score > 0]
    best = sorted(matching, key=lambda item: (-item[0], item[1]))[:MAX_MATCHES]
    return sorted(index for _, index in best)


def _merge_windows(indexes: list[int], total: int) -> list[tuple[int, int]]:
    """Context windows ``[start, end)`` around each index; overlapping or touching windows are merged."""
    windows: list[tuple[int, int]] = []
    for index in indexes:
        start, end = max(0, index - CONTEXT_LINES), min(total, index + CONTEXT_LINES + 1)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
    return windows


def _query_body(lines: list[Line], terms: list[str], query: str) -> str:
    indexes = _best_matches(lines, terms)
    if not indexes:
        start = truncate_text("\n".join(line.render() for line in lines), NO_MATCH_LIMIT)
        return f"No transcript line matches {query!r}; the beginning of the transcript follows.\n{start}"
    windows = _merge_windows(indexes, len(lines))
    shown = sum(end - start for start, end in windows)
    blocks = ["\n".join(line.render() for line in lines[start:end]) for start, end in windows]
    intro = f"Lines around the best matches for {query!r} ({shown} of {len(lines)} lines shown):"
    return intro + "\n" + "\n...\n".join(blocks)


def render_transcript(transcript: FetchedTranscript, query: str = "", limit: int = TRANSCRIPT_LIMIT) -> str:
    """Format ``transcript`` as ``[mm:ss] text`` lines, or only the lines around the matches of ``query``."""
    lines = _lines_of(transcript)
    if not lines:
        raise ToolError("the transcript has no text")
    terms = _query_terms(query)
    body = _query_body(lines, terms, query) if terms else "\n".join(line.render() for line in lines)
    text = f"{_header(transcript, lines)}\n{body}"
    if len(text) <= limit or terms:
        return truncate_text(text, limit)
    return f"{truncate_text(text, limit)}\nPass `query` to read only the lines about a topic."
