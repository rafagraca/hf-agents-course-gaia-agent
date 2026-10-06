"""Tests for ``transcribe_audio``: converting the formats the Groq API does not accept with ffmpeg.

The audio files are generated here (silence or placeholder bytes) and live in the temporary DATA
folder. HTTP is faked with ``responses``; the fake API key is never allowed to show up in a result.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
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


@pytest.fixture
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Pretend ffmpeg is installed; its 'conversion' writes placeholder mp3 bytes. Returns the commands it ran."""
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"ID3-converted-placeholder")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(shutil, "which", lambda name: "ffmpeg-for-tests" if name == "ffmpeg" else None)
    monkeypatch.setattr(subprocess, "run", fake_run)
    return commands


def make_audio(attachments: Path, name: str = "clip.mp3", content: bytes = b"fake-audio-bytes") -> str:
    (attachments / name).write_bytes(content)
    return name


# --------------------------------------------------------------------------- conversion with ffmpeg


def test_transcribe_audio_converts_formats_the_api_does_not_accept(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock, fake_ffmpeg: list[list[str]]
) -> None:
    name = make_audio(attachments, "voice.amr", b"original-amr-bytes")
    http.add(responses.POST, URL, body="converted ok", content_type="text/plain")

    assert transcribe(name) == "converted ok"

    command = fake_ffmpeg[0]
    assert command[0] == "ffmpeg-for-tests"
    assert str((attachments / name).resolve()) in command
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    body = http.calls[0].request.body.decode("latin-1")
    assert 'filename="audio.mp3"' in body
    assert "ID3-converted-placeholder" in body
    assert "original-amr-bytes" not in body


def test_transcribe_audio_converts_inside_the_data_folder_not_in_the_system_temp_folder(
    transcribe: TranscribeAudioTool,
    attachments: Path,
    http: responses.RequestsMock,
    fake_ffmpeg: list[list[str]],
    settings: Settings,
) -> None:
    name = make_audio(attachments, "voice.amr")
    http.add(responses.POST, URL, body="ok", content_type="text/plain")

    transcribe(name)

    assert Path(fake_ffmpeg[0][-1]).is_relative_to(settings.tmp_dir.resolve())


def test_ffmpeg_may_only_open_local_files_so_a_crafted_playlist_cannot_fetch_anything(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock, fake_ffmpeg: list[list[str]]
) -> None:
    name = make_audio(attachments, "voice.amr")
    http.add(responses.POST, URL, body="ok", content_type="text/plain")

    transcribe(name)

    command = fake_ffmpeg[0]
    assert command[command.index("-protocol_whitelist") + 1] == "file"
    assert command.index("-protocol_whitelist") < command.index("-i")


def test_transcribe_audio_cleans_up_its_temporary_files(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock, fake_ffmpeg: list[list[str]]
) -> None:
    name = make_audio(attachments, "voice.amr")
    http.add(responses.POST, URL, body="ok", content_type="text/plain")

    transcribe(name)

    assert not Path(fake_ffmpeg[0][-1]).exists()


def test_transcribe_audio_explains_that_ffmpeg_is_needed(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = make_audio(attachments, "voice.amr")
    monkeypatch.setattr(shutil, "which", lambda command: None)

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "ffmpeg" in result
    assert not http.calls


def test_transcribe_audio_reports_a_failed_conversion(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = make_audio(attachments, "voice.amr")
    monkeypatch.setattr(shutil, "which", lambda command: "ffmpeg-for-tests")

    def failed(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 1, b"", b"bad")

    monkeypatch.setattr(subprocess, "run", failed)

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "could not convert" in result
    assert not http.calls


def test_transcribe_audio_reports_an_ffmpeg_timeout(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = make_audio(attachments, "voice.amr")

    def slow(command: list[str], **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(shutil, "which", lambda command: "ffmpeg-for-tests")
    monkeypatch.setattr(subprocess, "run", slow)

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "too long" in result


def test_transcribe_audio_reports_an_ffmpeg_that_cannot_start(
    transcribe: TranscribeAudioTool, attachments: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = make_audio(attachments, "voice.amr")

    def broken(command: list[str], **kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(shutil, "which", lambda command: "ffmpeg-for-tests")
    monkeypatch.setattr(subprocess, "run", broken)

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "could not run ffmpeg" in result


def test_transcribe_audio_converts_and_retries_once_when_the_api_rejects_the_file(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock, fake_ffmpeg: list[list[str]]
) -> None:
    name = make_audio(attachments, "mislabelled.wav", b"not really a wav")

    def reply(request: requests.PreparedRequest) -> tuple[int, dict[str, str], str]:
        if b"ID3-converted-placeholder" in request.body:
            return 200, {"content-type": "text/plain"}, "recovered"
        return 400, {}, '{"error": {"message": "could not process file - is it a valid media file?"}}'

    http.add_callback(responses.POST, URL, callback=reply)

    assert transcribe(name) == "recovered"
    assert len(http.calls) == 2


def test_transcribe_audio_gives_up_when_the_converted_file_is_rejected_too(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock, fake_ffmpeg: list[list[str]]
) -> None:
    name = make_audio(attachments, "mislabelled.wav", b"not really a wav")
    http.add(responses.POST, URL, status=400, json={"error": {"message": "could not process file"}})

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "could not process" in result
    assert len(http.calls) == 2


def test_transcribe_audio_reports_a_rejected_file_when_ffmpeg_is_missing(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = make_audio(attachments, "mislabelled.wav", b"not really a wav")
    monkeypatch.setattr(shutil, "which", lambda command: None)
    http.add(responses.POST, URL, status=400, json={"error": {"message": "could not process file"}})

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "could not process" in result
    assert len(http.calls) == 1


def test_transcribe_audio_refuses_a_converted_file_that_is_still_too_large(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = make_audio(attachments, "voice.amr")

    def bloated(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        with Path(command[-1]).open("wb") as handle:
            handle.seek(25 * MEGABYTE)
            handle.write(b"\x00")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(shutil, "which", lambda command: "ffmpeg-for-tests")
    monkeypatch.setattr(subprocess, "run", bloated)

    result = transcribe(name)

    assert result.startswith("Error:")
    assert "25 MB" in result
    assert not http.calls


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_transcribe_audio_converts_with_the_real_ffmpeg(
    transcribe: TranscribeAudioTool, attachments: Path, http: responses.RequestsMock
) -> None:
    with wave.open(str(attachments / "tone.wav"), "wb") as writer:  # content is sniffed, the extension is not
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 16000)
    (attachments / "tone.wav").rename(attachments / "tone.unknownext")
    http.add(responses.POST, URL, body="real conversion", content_type="text/plain")

    assert transcribe("tone.unknownext") == "real conversion"

    body = http.calls[0].request.body
    assert b'filename="audio.mp3"' in body
    assert b"ID3" in body or b"\xff\xf3" in body or b"\xff\xfb" in body or b"\xff\xf2" in body
