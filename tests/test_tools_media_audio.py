"""Tests for ``transcribe_audio``: Groq Whisper over HTTP faked with ``responses`` (no network).

The audio files are generated here (silence or placeholder bytes) and live in the
temporary DATA folder. The fake API key is never allowed to show up in a result.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests
import responses

from gaia_agent.config import Settings
from gaia_agent.tools.media import TranscribeAudioTool

URL = "https://api.groq.com/openai/v1/audio/transcriptions"
FAKE_KEY = "fake-groq-key-for-tests"
MEGABYTE = 1024 * 1024


@pytest.fixture
def http():
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mocked:
        yield mocked


@pytest.fixture
def attachments(settings: Settings) -> Path:
    return settings.files_dir


@pytest.fixture
def transcribe(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> TranscribeAudioTool:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    return TranscribeAudioTool(settings)


def make_audio(attachments: Path, name: str = "clip.mp3", content: bytes = b"fake-audio-bytes") -> str:
    (attachments / name).write_bytes(content)
    return name


def uploaded_models(http: responses.RequestsMock) -> list[str]:
    models = []
    for call in http.calls:
        body = call.request.body.decode("latin-1")
        known = ("whisper-large-v3-turbo", "whisper-large-v3")
        models.append(next(name for name in known if f"\r\n\r\n{name}\r\n" in body))
    return models


# --------------------------------------------------------------------------- the request


def test_transcribe_audio_posts_the_file_with_the_documented_fields(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments, content=b"RIFF-pretend-audio")
    http.add(responses.POST, URL, body="Hello world", content_type="text/plain")

    result = transcribe(name)

    assert result == "Hello world"
    request = http.calls[0].request
    assert request.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    body = request.body.decode("latin-1")
    assert 'name="model"\r\n\r\nwhisper-large-v3-turbo' in body
    assert 'name="response_format"\r\n\r\ntext' in body
    assert 'name="file"; filename="audio.mp3"' in body
    assert "RIFF-pretend-audio" in body


def test_transcribe_audio_decodes_the_reply_as_utf8(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)
    http.add(responses.POST, URL, body="café ñ 日本\n".encode(), content_type="text/plain")

    assert transcribe(name) == "café ñ 日本"


def test_transcribe_audio_understands_a_json_reply(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)
    http.add(responses.POST, URL, json={"text": " from json "})

    assert transcribe(name) == "from json"


def test_transcribe_audio_keeps_the_raw_body_when_json_was_announced_but_not_sent(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)
    http.add(responses.POST, URL, body="just plain words", content_type="application/json")

    assert transcribe(name) == "just plain words"


def test_transcribe_audio_reports_silence(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)
    http.add(responses.POST, URL, body="  \n", content_type="text/plain")

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "no speech" in result


def test_transcribe_audio_truncates_a_very_long_transcript(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)
    http.add(responses.POST, URL, body="word " * 5000, content_type="text/plain")

    result = transcribe(name)

    assert "[truncated:" in result
    assert len(result) < 6200


@pytest.mark.parametrize("name", ["voice.mp3", "talk.WAV", "memo.m4a", "song.flac", "clip.ogg", "sound.webm"])
def test_transcribe_audio_uploads_accepted_formats_without_conversion(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock, name: str
) -> None:
    make_audio(attachments, name)
    http.add(responses.POST, URL, body="ok", content_type="text/plain")

    assert transcribe(name) == "ok"
    assert f'filename="audio{Path(name).suffix.lower()}"' in http.calls[0].request.body.decode("latin-1")


# --------------------------------------------------------------------------- what is refused


def test_transcribe_audio_refuses_a_file_outside_data(
    transcribe: TranscribeAudioTool, tmp_path: Path, http: responses.RequestsMock
) -> None:
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"private audio")

    result = transcribe(str(outside))

    assert result.startswith("Error:")
    assert "outside" in result
    assert not http.calls


def test_transcribe_audio_refuses_files_over_25_megabytes(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    with (attachments / "huge.mp3").open("wb") as handle:
        handle.seek(25 * MEGABYTE)
        handle.write(b"\x00")

    result = transcribe("huge.mp3")

    assert result.startswith("Error:")
    assert "25 MB" in result
    assert not http.calls


def test_transcribe_audio_accepts_a_file_of_exactly_25_megabytes(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    with (attachments / "edge.mp3").open("wb") as handle:
        handle.seek(25 * MEGABYTE - 1)
        handle.write(b"\x00")
    http.add(responses.POST, URL, body="fits", content_type="text/plain")

    assert transcribe("edge.mp3") == "fits"


def test_transcribe_audio_refuses_an_empty_file(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments, content=b"")

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "empty" in result
    assert not http.calls


def test_transcribe_audio_reports_a_missing_file(transcribe: TranscribeAudioTool) -> None:
    assert transcribe("absent.mp3").startswith("Error: file not found")


def test_transcribe_audio_without_a_key_says_so_and_sends_nothing(
    settings: Settings, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)

    result = TranscribeAudioTool(settings)(name)

    assert result.startswith("Error:")
    assert "GROQ_API_KEY" in result
    assert not http.calls


def test_transcribe_audio_works_without_explicit_settings(
    attachments: Path, http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    name = make_audio(attachments)
    http.add(responses.POST, URL, body="default settings", content_type="text/plain")

    assert TranscribeAudioTool()(name) == "default settings"


# --------------------------------------------------------------------------- API failures


def test_transcribe_audio_falls_back_to_the_second_model_when_rate_limited(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)

    def reply(request: requests.PreparedRequest) -> tuple[int, dict[str, str], str]:
        if b"whisper-large-v3-turbo" in request.body:
            return 429, {"retry-after": "7"}, '{"error": {"message": "slow down"}}'
        return 200, {"content-type": "text/plain"}, "from the plan b model"

    http.add_callback(responses.POST, URL, callback=reply)

    assert transcribe(name) == "from the plan b model"
    assert uploaded_models(http) == ["whisper-large-v3-turbo", "whisper-large-v3"]


@pytest.mark.parametrize("status", [404, 500, 503])
def test_transcribe_audio_falls_back_when_the_first_model_is_unavailable(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock, status: int
) -> None:
    name = make_audio(attachments)
    http.add(responses.POST, URL, status=status, json={"error": {"message": "unavailable"}})
    http.add(responses.POST, URL, body="second model", content_type="text/plain")

    assert transcribe(name) == "second model"


def test_transcribe_audio_falls_back_when_the_first_model_is_rejected_by_name(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)
    decommissioned = {"error": {"message": "The model `whisper-large-v3-turbo` has been decommissioned"}}
    http.add(responses.POST, URL, status=400, json=decommissioned)
    http.add(responses.POST, URL, body="second model", content_type="text/plain")

    assert transcribe(name) == "second model"


def test_transcribe_audio_reports_a_rate_limit_on_every_model(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)
    http.add(responses.POST, URL, status=429, headers={"retry-after": "12"}, json={"error": {"message": "limit"}})
    http.add(responses.POST, URL, status=429, headers={"retry-after": "30"}, json={"error": {"message": "limit"}})

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "whisper-large-v3-turbo" in result
    assert "whisper-large-v3:" in result
    assert "rate limit" in result
    assert "retry after 12 s" in result


def test_transcribe_audio_does_not_retry_when_the_key_is_rejected(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)
    http.add(responses.POST, URL, status=401, json={"error": {"message": f"Invalid API Key {FAKE_KEY}"}})

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "GROQ_API_KEY" in result
    assert FAKE_KEY not in result
    assert len(http.calls) == 1


def test_transcribe_audio_reports_a_payload_that_is_too_large(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)
    http.add(responses.POST, URL, status=413, json={"error": {"message": "too big"}})

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "too large" in result
    assert len(http.calls) == 1


def test_transcribe_audio_never_echoes_the_key_from_an_error_body(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)
    http.add(responses.POST, URL, status=418, body=f"teapot says {FAKE_KEY}", content_type="text/plain")

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "HTTP 418" in result
    assert FAKE_KEY not in result
    assert "[redacted]" in result


def test_transcribe_audio_reports_a_timeout(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)
    http.add(responses.POST, URL, body=requests.Timeout("read timed out"))

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "timed out" in result


def test_transcribe_audio_reports_a_file_that_cannot_be_opened(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = make_audio(attachments)
    real_open = Path.open

    def locked(self: Path, *args: object, **kwargs: object):
        if self.name == name:
            raise PermissionError("locked by another process")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", locked)

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "could not read the audio file" in result
    assert "PermissionError" in result
    assert not http.calls


def test_transcribe_audio_reports_a_connection_failure(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    name = make_audio(attachments)
    http.add(responses.POST, URL, body=requests.ConnectionError(f"cannot connect with {FAKE_KEY}"))

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "could not reach" in result
    assert FAKE_KEY not in result
