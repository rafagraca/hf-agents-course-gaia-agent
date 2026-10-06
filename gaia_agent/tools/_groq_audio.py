"""Speech-to-text through Groq's Whisper endpoint (OpenAI-compatible multipart upload).

``whisper-large-v3-turbo`` is tried first and ``whisper-large-v3`` is the plan B
(the two have separate rate limits). A format the API does not accept is
converted to a small mono mp3 with ffmpeg when ffmpeg is installed. Failures
become short :class:`ToolError` messages that never contain the API key.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import requests

from gaia_agent.tools._api_errors import error_message
from gaia_agent.tools._local_files import ToolError, scratch_folder

TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
TRANSCRIPTION_MODELS = ("whisper-large-v3-turbo", "whisper-large-v3")
ACCEPTED_SUFFIXES = frozenset({".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"})
MEGABYTE = 1024 * 1024
MAX_UPLOAD_BYTES = 25 * MEGABYTE
REQUEST_TIMEOUT_S = (10, 90)  # connect, read
FFMPEG_TIMEOUT_S = 120


class _TryNextModel(ToolError):
    """This model cannot serve the request right now; the next one may."""


class _FileRejected(ToolError):
    """The API could not process the audio itself (damaged or mislabelled file)."""


def _check_size(path: Path) -> None:
    size = path.stat().st_size
    if size == 0:
        raise ToolError("the audio file is empty")
    if size > MAX_UPLOAD_BYTES:
        raise ToolError(f"the audio file is {size / MEGABYTE:.1f} MB; the transcription limit is 25 MB")


def convert_to_mp3(source: Path, folder: Path) -> Path:
    """Convert any audio or video file ffmpeg understands to a speech-sized mono 16 kHz mp3 inside ``folder``."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ToolError(
            "this audio format is not accepted by the transcription service and ffmpeg is not installed to convert it"
        )
    target = folder / "audio.mp3"
    command = [
        # A crafted "audio" file can be a playlist that points ffmpeg at other files or at the network:
        # only plain local files may be opened.
        *(ffmpeg, "-nostdin", "-y", "-v", "error", "-protocol_whitelist", "file", "-i", str(source)),
        *("-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(target)),
    ]
    try:
        done = subprocess.run(command, capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ToolError("converting the audio with ffmpeg took too long") from exc
    except OSError as exc:
        raise ToolError(f"could not run ffmpeg ({type(exc).__name__})") from exc
    if done.returncode != 0 or not target.is_file():
        raise ToolError("ffmpeg could not convert this file to mp3 (is it a valid audio or video file?)")
    return target


def _retry_hint(response: requests.Response) -> str:
    seconds = response.headers.get("retry-after", "")
    return f", retry after {seconds} s" if re.fullmatch(r"\d+(\.\d+)?", seconds) else ""


def _failure_for(response: requests.Response, model: str, api_key: str) -> ToolError:
    status = response.status_code
    if status in (401, 403):
        return ToolError(f"Groq rejected the API key (HTTP {status}); check GROQ_API_KEY")
    if status == 413:
        return ToolError("the audio file is too large for the Groq API")
    if status == 429:
        return _TryNextModel(f"{model}: rate limit reached (HTTP 429{_retry_hint(response)})")
    message = error_message(response, api_key)
    if status in (404, 408, 409) or status >= 500 or (status == 400 and "model" in message.lower()):
        return _TryNextModel(f"{model}: {message or 'service unavailable'} (HTTP {status})")
    if status in (400, 415, 422):
        return _FileRejected(f"Groq could not process the file: {message}")
    return ToolError(f"Groq API error (HTTP {status}): {message}")


def _json_text(body: str) -> str | None:
    try:
        return str(json.loads(body)["text"]).strip()
    except (ValueError, KeyError, TypeError):
        return None


def _reply_text(response: requests.Response) -> str:
    """The transcript in a successful reply: plain text, or the ``text`` field when the server sent JSON."""
    body = response.content.decode("utf-8", errors="replace").strip()  # never trust a missing charset
    if "json" in response.headers.get("content-type", "").lower():
        parsed = _json_text(body)
        body = body if parsed is None else parsed
    if not body:
        raise ToolError("no speech was detected in the audio")
    return body


def _post(path: Path, api_key: str, model: str, filename: str) -> str:
    try:
        with path.open("rb") as handle:
            response = requests.post(
                TRANSCRIPTION_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                data={"model": model, "response_format": "text"},
                files={"file": (filename, handle)},
                timeout=REQUEST_TIMEOUT_S,
            )
    except requests.Timeout as exc:
        raise ToolError("the Groq request timed out") from exc
    except requests.RequestException as exc:
        raise ToolError("could not reach the Groq API") from exc
    except OSError as exc:  # after the requests errors, which are OSErrors too
        raise ToolError(f"could not read the audio file ({type(exc).__name__})") from exc
    if response.status_code != 200:
        raise _failure_for(response, model, api_key)
    return _reply_text(response)


def _transcribe(upload: Path, api_key: str) -> str:
    """Try each model in turn; stop at the first that answers or at an error no other model could fix."""
    _check_size(upload)
    filename = f"audio{upload.suffix.lower()}"  # a plain ASCII name: the original may hold any character
    problems: list[str] = []
    for model in TRANSCRIPTION_MODELS:
        try:
            return _post(upload, api_key, model, filename)
        except _TryNextModel as exc:
            problems.append(str(exc))
    raise ToolError("; ".join(problems))


def transcribe_file(path: Path, api_key: str, scratch_dir: Path | None = None) -> str:
    """Transcribe the audio file at ``path``; raise :class:`ToolError` with a safe message on any failure.

    A converted copy of the audio, when one is needed, lives in a throw-away folder inside ``scratch_dir``
    (the system temp folder when None).
    """
    _check_size(path)
    with scratch_folder(scratch_dir, "gaia-audio-") as folder:
        if path.suffix.lower() not in ACCEPTED_SUFFIXES:
            return _transcribe(convert_to_mp3(path, folder), api_key)
        try:
            return _transcribe(path, api_key)
        except _FileRejected as exc:
            if shutil.which("ffmpeg") is None:
                raise ToolError(str(exc)) from exc
            return _transcribe(convert_to_mp3(path, folder), api_key)
