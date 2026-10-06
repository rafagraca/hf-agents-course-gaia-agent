"""Tests for ``gaia_agent.tools.media``: YouTube transcripts and the Gemini-backed image/video tools.

No test touches the network. The real ``youtube-transcript-api`` library runs
against HTTP responses faked with ``responses``; the Gemini REST endpoint is faked
the same way. All transcripts, videos and images are invented.
"""

from __future__ import annotations

import pytest
import responses
from youtube_transcript_api import FetchedTranscript, FetchedTranscriptSnippet

from gaia_agent.tools import media
from gaia_agent.tools._youtube import canonical_url, parse_video_id, render_transcript
from gaia_agent.tools.files import ReadFileTool, RunPythonFileTool
from gaia_agent.tools.media import (
    AskAboutVideoTool,
    DescribeImageTool,
    TranscribeAudioTool,
    YouTubeTranscriptTool,
    YoutubeTranscriptTool,
)

# The tool list is part of the system prompt, which is sent on every step: keep these six tools terse.
RENDERED_TOOLS_BUDGET_CHARS = 1400

VIDEO_ID = "abcDEF12345"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


@pytest.fixture
def http():
    """Fake HTTP layer for ``requests``: any request that is not registered fails the test."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mocked:
        yield mocked


def snippets(*items: tuple[float, str]) -> FetchedTranscript:
    return FetchedTranscript(
        snippets=[FetchedTranscriptSnippet(text=text, start=start, duration=2.0) for start, text in items],
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=False,
    )


# --------------------------------------------------------------------------- tool schemas


def test_the_media_and_file_tools_render_compactly_in_the_prompt() -> None:
    tools = [
        YoutubeTranscriptTool(),
        TranscribeAudioTool(),
        DescribeImageTool(),
        AskAboutVideoTool(),
        ReadFileTool(),
        RunPythonFileTool(),
    ]

    rendered = sum(len(tool.to_code_prompt()) for tool in tools)

    assert rendered <= RENDERED_TOOLS_BUDGET_CHARS


def test_the_spec_spelling_of_the_youtube_tool_resolves_to_the_same_class() -> None:
    assert YouTubeTranscriptTool is YoutubeTranscriptTool


def test_the_spec_spelling_is_not_listed_as_a_second_tool_class() -> None:
    # Code that scans the module for Tool subclasses must not register youtube_transcript twice.
    assert "YouTubeTranscriptTool" not in dir(media)
    assert "YoutubeTranscriptTool" in dir(media)


def test_the_media_module_rejects_unknown_attributes() -> None:
    with pytest.raises(AttributeError):
        _ = media.NoSuchTool


# --------------------------------------------------------------------------- video ids


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtube.com/watch?v={VIDEO_ID}&t=42s",
        f"https://m.youtube.com/watch?feature=share&v={VIDEO_ID}",
        f"https://music.youtube.com/watch?v={VIDEO_ID}&list=PL123",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}?si=tracking&t=10",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}?rel=0",
        f"https://www.youtube-nocookie.com/embed/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}?feature=share",
        f"https://www.youtube.com/v/{VIDEO_ID}",
        f"youtube.com/watch?v={VIDEO_ID}",
        f"www.youtu.be/{VIDEO_ID}",
        f"  https://youtu.be/{VIDEO_ID}  ",
        f'"https://youtu.be/{VIDEO_ID}"',
        VIDEO_ID,
    ],
)
def test_parse_video_id_understands_every_url_format(url: str) -> None:
    assert parse_video_id(url) == VIDEO_ID


def test_parse_video_id_accepts_ids_made_of_dashes_and_underscores() -> None:
    assert parse_video_id("https://youtu.be/-_-_-_-_-_-") == "-_-_-_-_-_-"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not a url at all",
        "https://example.com/watch?v=abcDEF12345",
        "https://evilyoutube.com/watch?v=abcDEF12345",
        "https://youtube.com.evil.example/watch?v=abcDEF12345",
        "https://www.youtube.com/",
        "https://www.youtube.com/watch?v=",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/watch?v=abcDEF123456",
        "https://www.youtube.com/channel/UCabcdefghijk",
        "https://www.youtube.com/shorts/",
        "https://[unbalanced",
        "http://[::1",
        None,
        42,
    ],
)
def test_parse_video_id_rejects_everything_else(url: object) -> None:
    assert parse_video_id(url) is None


def test_canonical_url_drops_tracking_parameters() -> None:
    assert canonical_url(VIDEO_ID) == WATCH_URL


# --------------------------------------------------------------------------- rendering


def test_render_transcript_prefixes_every_line_with_minutes_and_seconds() -> None:
    transcript = snippets((0.4, "Hello there"), (65.9, "a minute later"), (3725.0, "much later"))

    text = render_transcript(transcript)

    assert "[00:00] Hello there" in text
    assert "[01:05] a minute later" in text
    assert "[62:05] much later" in text


def test_render_transcript_collapses_whitespace_and_skips_blank_snippets() -> None:
    transcript = snippets((0.0, "two\nlines   here"), (2.0, "   "), (4.0, "end"))

    text = render_transcript(transcript)

    assert "[00:00] two lines here" in text
    assert text.count("[") == 2


def test_render_transcript_header_names_the_language_and_kind() -> None:
    text = render_transcript(snippets((0.0, "Hello")))

    assert text.splitlines()[0].startswith("YouTube transcript")
    assert "en" in text.splitlines()[0]
    assert "manual" in text.splitlines()[0]


def test_render_transcript_truncates_long_transcripts_and_suggests_a_query() -> None:
    transcript = snippets(*[(float(i), f"line number {i} " + "word " * 10) for i in range(400)])

    text = render_transcript(transcript)

    assert "[truncated:" in text
    assert "query" in text
    assert len(text) < 6500


def test_render_transcript_with_query_does_not_suggest_a_query_when_it_has_to_truncate() -> None:
    transcript = snippets(*[(float(i * 5), "needle in a very long caption " + "word " * 150) for i in range(40)])

    text = render_transcript(transcript, query="needle")

    assert "[truncated:" in text
    assert "Pass `query`" not in text


def test_render_transcript_with_query_returns_the_lines_around_the_matches() -> None:
    transcript = snippets(*[(float(i * 5), f"filler sentence {i}") for i in range(60)])
    transcript.snippets[30] = FetchedTranscriptSnippet(text="the rare Dromedary appears", start=150.0, duration=2.0)

    text = render_transcript(transcript, query="dromedary")

    assert "[02:30] the rare Dromedary appears" in text
    assert "[02:20] filler sentence 28" in text
    assert "[02:40] filler sentence 32" in text
    assert "filler sentence 5" not in text
    assert "filler sentence 55" not in text


def test_render_transcript_with_query_separates_distant_matches() -> None:
    items = [(float(i * 5), f"filler sentence {i}") for i in range(80)]
    items[10] = (50.0, "first mention of marmots")
    items[70] = (350.0, "second mention of marmots")

    text = render_transcript(snippets(*items), query="marmots")

    assert "first mention of marmots" in text
    assert "second mention of marmots" in text
    assert "\n...\n" in text


def test_render_transcript_with_query_prefers_lines_matching_more_terms() -> None:
    items = [(float(i * 5), f"filler sentence {i}") for i in range(120)]
    for index in range(3, 100, 10):  # ten weak matches, far apart
        items[index] = (float(index * 5), f"red mention {index}")
    items[55] = (275.0, "a red bicycle with a blue basket")

    text = render_transcript(snippets(*items), query="red blue basket bicycle")

    assert "a red bicycle with a blue basket" in text
    assert "red mention 3" in text
    assert "red mention 93" not in text  # only the best few matches are kept


def test_render_transcript_with_query_ignores_stop_words_and_case() -> None:
    items = [(float(i * 5), f"filler sentence {i}") for i in range(30)]
    items[12] = (60.0, "The Violin sounded")

    text = render_transcript(snippets(*items), query="What is the VIOLIN?")

    assert "The Violin sounded" in text
    assert "filler sentence 0" not in text


def test_render_transcript_with_a_query_without_a_match_says_so_and_shows_the_start() -> None:
    transcript = snippets((0.0, "welcome to the show"), (3.0, "today we talk about gardens"))

    text = render_transcript(transcript, query="submarine")

    assert "No transcript line matches" in text
    assert "welcome to the show" in text


def test_render_transcript_with_only_stop_words_in_the_query_still_matches() -> None:
    transcript = snippets((0.0, "to be or not to be"), (3.0, "that is the question"))

    assert "to be or not to be" in render_transcript(transcript, query="to be")


def test_render_transcript_keeps_numeric_query_terms() -> None:
    items = [(float(i * 5), f"filler sentence {i}") for i in range(30)]
    items[20] = (100.0, "there were 42 of them")

    assert "there were 42 of them" in render_transcript(snippets(*items), query="how many 42")
