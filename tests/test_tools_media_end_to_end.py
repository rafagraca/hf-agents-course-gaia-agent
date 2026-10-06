"""Tests for the ``youtube_transcript`` tool end to end (HTTP and the transcript library faked)."""

from __future__ import annotations

import pytest
import requests
import responses
from youtube_transcript_api import FetchedTranscript, FetchedTranscriptSnippet

from gaia_agent.tools._youtube import YOUTUBE_TIMEOUT_S
from gaia_agent.tools.media import (
    YoutubeTranscriptTool,
)

# The tool list is part of the system prompt, which is sent on every step: keep these six tools terse.

VIDEO_ID = "abcDEF12345"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
INNERTUBE_KEY = "test-innertube-key"
PLAYER_URL = f"https://www.youtube.com/youtubei/v1/player?key={INNERTUBE_KEY}"
WATCH_PAGE = f'<html><script>var cfg = {{"INNERTUBE_API_KEY":"{INNERTUBE_KEY}"}};</script></html>'


@pytest.fixture
def http():
    """Fake HTTP layer for ``requests``: any request that is not registered fails the test."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mocked:
        yield mocked


def timedtext(*snippets: tuple[float, float, str]) -> str:
    """The XML YouTube serves for a caption track."""
    items = "".join(f'<text start="{start}" dur="{dur}">{text}</text>' for start, dur, text in snippets)
    return f'<?xml version="1.0" encoding="utf-8" ?><transcript>{items}</transcript>'


def caption_track(code: str, name: str, *, generated: bool = False) -> dict:
    track = {
        "baseUrl": f"https://www.youtube.com/api/timedtext/{code}",
        "name": {"runs": [{"text": name}]},
        "languageCode": code,
    }
    if generated:
        track["kind"] = "asr"
    return track


def mock_video(http: responses.RequestsMock, tracks: dict[str, str | None], *, details: dict[str, dict]) -> None:
    """Fake the three requests the library makes: watch page, player data and one request per caption track."""
    http.add(responses.GET, WATCH_URL, body=WATCH_PAGE)
    player = {
        "playabilityStatus": {"status": "OK"},
        "captions": {"playerCaptionsTracklistRenderer": {"captionTracks": list(details.values())}},
    }
    http.add(responses.POST, PLAYER_URL, json=player)
    for code, xml in tracks.items():
        http.add(responses.GET, details[code]["baseUrl"], body=xml)


def snippets(*items: tuple[float, str]) -> FetchedTranscript:
    return FetchedTranscript(
        snippets=[FetchedTranscriptSnippet(text=text, start=start, duration=2.0) for start, text in items],
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=False,
    )


# --------------------------------------------------------------------------- the tool end to end


def test_youtube_transcript_tool_returns_timestamped_lines(http: responses.RequestsMock) -> None:
    track = caption_track("en", "English (auto-generated)", generated=True)
    xml = timedtext((0.5, 2.0, "Hello &amp; welcome"), (2.5, 3.0, "it&amp;#39;s a fine day"), (70.0, 2.0, "bye"))
    mock_video(http, {"en": xml}, details={"en": track})

    result = YoutubeTranscriptTool()(WATCH_URL)

    assert "[00:00] Hello & welcome" in result
    assert "[00:02] it's a fine day" in result
    assert "[01:10] bye" in result
    assert "auto-generated" in result.splitlines()[0]


def test_youtube_transcript_tool_never_waits_on_youtube_without_a_timeout(http: responses.RequestsMock) -> None:
    track = caption_track("en", "English")
    mock_video(http, {"en": timedtext((0.0, 1.0, "quick"))}, details={"en": track})

    YoutubeTranscriptTool()(WATCH_URL)

    assert len(http.calls) == 3  # watch page, player data, caption track
    assert all(call.request.req_kwargs["timeout"] == YOUTUBE_TIMEOUT_S for call in http.calls)


def test_youtube_transcript_tool_accepts_short_links_and_none_query(http: responses.RequestsMock) -> None:
    track = caption_track("en", "English")
    mock_video(http, {"en": timedtext((0.0, 1.0, "short link works"))}, details={"en": track})

    result = YoutubeTranscriptTool()(f"https://youtu.be/{VIDEO_ID}?t=3", None)

    assert "short link works" in result


def test_youtube_transcript_tool_with_query_returns_only_the_relevant_part(http: responses.RequestsMock) -> None:
    track = caption_track("en", "English")
    items = [(float(i * 4), 2.0, f"filler sentence {i}") for i in range(100)]
    items[50] = (200.0, 2.0, "the heron lands")
    mock_video(http, {"en": timedtext(*items)}, details={"en": track})

    result = YoutubeTranscriptTool()(WATCH_URL, "heron")

    assert "[03:20] the heron lands" in result
    assert "filler sentence 5\n" not in result


def test_youtube_transcript_tool_prefers_a_manual_english_track_over_an_automatic_one(
    http: responses.RequestsMock,
) -> None:
    auto = caption_track("en", "English (auto-generated)", generated=True)
    manual = caption_track("en-GB", "English (UK)")
    mock_video(
        http,
        {"en": timedtext((0.0, 1.0, "automatic words")), "en-GB": timedtext((0.0, 1.0, "careful manual words"))},
        details={"en": auto, "en-GB": manual},
    )

    result = YoutubeTranscriptTool()(WATCH_URL)

    assert "careful manual words" in result
    assert "automatic words" not in result


def test_youtube_transcript_tool_falls_back_to_another_language(http: responses.RequestsMock) -> None:
    track = caption_track("es", "Spanish")
    mock_video(http, {"es": timedtext((0.0, 2.0, "hola a todos"))}, details={"es": track})

    result = YoutubeTranscriptTool()(WATCH_URL)

    assert "hola a todos" in result
    assert "es" in result.splitlines()[0]
    assert "not English" in result.splitlines()[0]


def test_youtube_transcript_tool_says_clearly_when_there_are_no_captions(http: responses.RequestsMock) -> None:
    http.add(responses.GET, WATCH_URL, body=WATCH_PAGE)
    http.add(responses.POST, PLAYER_URL, json={"playabilityStatus": {"status": "OK"}})

    result = YoutubeTranscriptTool()(WATCH_URL)

    assert result.startswith("Error:")
    assert "no transcript" in result.lower()
    assert "do not guess" in result.lower()


def test_youtube_transcript_tool_reports_an_unavailable_video(http: responses.RequestsMock) -> None:
    http.add(responses.GET, WATCH_URL, body=WATCH_PAGE)
    status = {"status": "ERROR", "reason": "This video is unavailable"}
    http.add(responses.POST, PLAYER_URL, json={"playabilityStatus": status})

    result = YoutubeTranscriptTool()(WATCH_URL)

    assert result.startswith("Error:")
    assert "unavailable" in result


def test_youtube_transcript_tool_reports_an_age_restricted_video(http: responses.RequestsMock) -> None:
    http.add(responses.GET, WATCH_URL, body=WATCH_PAGE)
    status = {"status": "LOGIN_REQUIRED", "reason": "This video may be inappropriate for some users."}
    http.add(responses.POST, PLAYER_URL, json={"playabilityStatus": status})

    assert "age-restricted" in YoutubeTranscriptTool()(WATCH_URL)


def test_youtube_transcript_tool_reports_an_ip_block(http: responses.RequestsMock) -> None:
    http.add(responses.GET, WATCH_URL, status=429)

    result = YoutubeTranscriptTool()(WATCH_URL)

    assert result.startswith("Error:")
    assert "blocking" in result


def test_youtube_transcript_tool_reports_unplayable_videos_without_the_library_boilerplate(
    http: responses.RequestsMock,
) -> None:
    http.add(responses.GET, WATCH_URL, body=WATCH_PAGE)
    status = {"status": "UNPLAYABLE", "reason": "Video unavailable in your country"}
    http.add(responses.POST, PLAYER_URL, json={"playabilityStatus": status})

    result = YoutubeTranscriptTool()(WATCH_URL)

    assert result.startswith("Error:")
    assert "github" not in result.lower()
    assert len(result) < 400


def test_youtube_transcript_tool_names_the_error_type_of_an_unexpected_failure(http: responses.RequestsMock) -> None:
    http.add(responses.GET, WATCH_URL, status=500)

    result = YoutubeTranscriptTool()(WATCH_URL)

    assert result.startswith("Error:")
    assert "YouTubeRequestFailed" in result
    assert "github" not in result.lower()


def test_youtube_transcript_tool_reports_a_video_whose_caption_list_is_empty(http: responses.RequestsMock) -> None:
    mock_video(http, {}, details={})

    result = YoutubeTranscriptTool()(WATCH_URL)

    assert result.startswith("Error:")
    assert "no transcript" in result


def test_youtube_transcript_tool_reports_a_network_failure(http: responses.RequestsMock) -> None:
    http.add(responses.GET, WATCH_URL, body=requests.ConnectionError("connection refused"))

    result = YoutubeTranscriptTool()(WATCH_URL)

    assert result.startswith("Error:")
    assert "could not reach YouTube" in result


def test_youtube_transcript_tool_reports_an_empty_caption_file(http: responses.RequestsMock) -> None:
    track = caption_track("en", "English")
    mock_video(http, {"en": ""}, details={"en": track})

    result = YoutubeTranscriptTool()(WATCH_URL)

    assert result.startswith("Error:")
    assert "empty or unreadable" in result


def test_youtube_transcript_tool_reports_a_transcript_without_text(http: responses.RequestsMock) -> None:
    track = caption_track("en", "English")
    mock_video(http, {"en": timedtext((0.0, 1.0, "  "))}, details={"en": track})

    result = YoutubeTranscriptTool()(WATCH_URL)

    assert result.startswith("Error:")
    assert "no text" in result


@pytest.mark.parametrize("url", ["https://example.com/video", "", "just words"])
def test_youtube_transcript_tool_rejects_urls_that_are_not_youtube_videos(url: str) -> None:
    result = YoutubeTranscriptTool()(url)

    assert result.startswith("Error:")
    assert "YouTube" in result
