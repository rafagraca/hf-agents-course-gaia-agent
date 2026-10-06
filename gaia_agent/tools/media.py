"""Audio, image and video tools.

``youtube_transcript`` and ``transcribe_audio`` are always available. ``describe_image``
(Groq vision, needs ``GROQ_API_KEY``) and the Gemini-backed ``ask_about_video`` (needs
``GEMINI_API_KEY``) are offered only when their key exists, through :func:`optional_media_tools`.

The tool descriptions are deliberately terse: they are sent to the model on every
step, so each word costs tokens. Expected failures come back as a short
``Error: ...`` string the agent can react to; no secret ever appears in one.
"""

from __future__ import annotations

from smolagents import Tool

from gaia_agent.config import Settings, get_secret
from gaia_agent.tools._gemini import ask_gemini, gemini_key, question_prompt, video_part
from gaia_agent.tools._groq_audio import transcribe_file
from gaia_agent.tools._groq_vision import ask_groq_about_image
from gaia_agent.tools._local_files import AttachmentSandbox, ToolError, truncate_text
from gaia_agent.tools._youtube import fetch_transcript, parse_video_id, render_transcript

TRANSCRIPT_LIMIT = 6000
ANSWER_LIMIT = 4000


class YoutubeTranscriptTool(Tool):
    """Fetch the transcript of a YouTube video as ``[mm:ss] text`` lines."""

    name = "youtube_transcript"
    description = "Get a YouTube video's transcript as [mm:ss] lines. `query` keeps only the lines around matches."
    inputs = {
        "url": {"type": "string", "description": "YouTube video URL."},
        "query": {"type": "string", "description": "Words to look for (optional).", "nullable": True},
    }
    output_type = "string"

    def forward(self, url: str, query: str = "") -> str:
        video_id = parse_video_id(url)
        if video_id is None:
            return f"Error: not a YouTube video URL: {str(url)[:200]}"
        try:
            return render_transcript(fetch_transcript(video_id), query or "")
        except ToolError as exc:
            return f"Error: {exc}. Do not guess what the video says."


class TranscribeAudioTool(Tool):
    """Transcribe an audio attachment to text with Groq Whisper (needs ``GROQ_API_KEY``).

    ``settings`` fixes the DATA folder the tool may read from; without it the folder
    comes from the environment (``GAIA_DATA_DIR``) when the tool is first used.
    """

    name = "transcribe_audio"
    description = "Transcribe an audio attachment to text."
    inputs = {"file_path": {"type": "string", "description": "Attachment file name or path."}}
    output_type = "string"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self._settings = settings

    def forward(self, file_path: str) -> str:
        try:
            sandbox = AttachmentSandbox.from_settings(self._settings)
            path = sandbox.resolve_file(file_path)
            api_key = get_secret("GROQ_API_KEY")
            if api_key is None:
                raise ToolError("GROQ_API_KEY is not set, so audio cannot be transcribed")
            return truncate_text(transcribe_file(path, api_key, sandbox.scratch_dir), TRANSCRIPT_LIMIT)
        except ToolError as exc:
            return f"Error: {exc}"


class DescribeImageTool(Tool):
    """Answer a question about an image attachment with Groq vision (needs ``GROQ_API_KEY``)."""

    name = "describe_image"
    description = "Answer a question about an image attachment."
    inputs = {
        "file_path": {"type": "string", "description": "Attachment file name or path."},
        "question": {"type": "string", "description": "What to find out."},
    }
    output_type = "string"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self._settings = settings

    def forward(self, file_path: str, question: str) -> str:
        try:
            path = AttachmentSandbox.from_settings(self._settings).resolve_file(file_path)
            api_key = get_secret("GROQ_API_KEY")
            if api_key is None:
                raise ToolError("GROQ_API_KEY is not set, so images cannot be described")
            return truncate_text(ask_groq_about_image(api_key, path, question), ANSWER_LIMIT)
        except ToolError as exc:
            return f"Error: {exc}"


class AskAboutVideoTool(Tool):
    """Answer a question about a public YouTube video with Gemini (needs ``GEMINI_API_KEY``)."""

    name = "ask_about_video"
    description = "Answer a question about a YouTube video."
    inputs = {
        "url": {"type": "string", "description": "YouTube video URL."},
        "question": {"type": "string", "description": "What to find out."},
    }
    output_type = "string"

    def forward(self, url: str, question: str) -> str:
        try:
            prompt = question_prompt("video", question)
            media = video_part(url)
            api_key = gemini_key()
            return truncate_text(ask_gemini(api_key, media, prompt), ANSWER_LIMIT)
        except ToolError as exc:
            return f"Error: {exc}"


def optional_media_tools(settings: Settings) -> list[Tool]:
    """Return ``describe_image`` when ``GROQ_API_KEY`` is set and ``ask_about_video`` when ``GEMINI_API_KEY`` is."""
    tools: list[Tool] = []
    if get_secret("GROQ_API_KEY") is not None:
        tools.append(DescribeImageTool(settings))
    if get_secret("GEMINI_API_KEY") is not None:
        tools.append(AskAboutVideoTool())
    return tools


def __getattr__(name: str) -> object:
    """Resolve ``YouTubeTranscriptTool``, the spelling used in the project spec, to :class:`YoutubeTranscriptTool`.

    It is a module-level ``__getattr__`` rather than a second assignment so that code scanning the module for
    ``Tool`` subclasses sees the class once and never registers ``youtube_transcript`` twice.
    """
    if name == "YouTubeTranscriptTool":
        return YoutubeTranscriptTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
