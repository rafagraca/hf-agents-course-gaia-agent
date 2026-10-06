"""Tools exposed to the agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gaia_agent.config import get_secret, run_python_enabled
from gaia_agent.tools.deep_search import DeepSearchTool
from gaia_agent.tools.files import ReadFileTool, RunPythonFileTool
from gaia_agent.tools.media import TranscribeAudioTool, YoutubeTranscriptTool, optional_media_tools
from gaia_agent.tools.web import ReadWebpageTool, WebSearchTool
from gaia_agent.tools.wikipedia import WikipediaPageTool, WikipediaSearchTool

if TYPE_CHECKING:
    from smolagents import Tool

    from gaia_agent.config import Settings

__all__ = ["default_tools"]


def default_tools(settings: Settings) -> list[Tool]:
    """Return the agent's tools: web, Wikipedia, files, audio and YouTube transcripts.

    ``describe_image`` is added only when ``GROQ_API_KEY`` is available and
    ``ask_about_video`` only when ``GEMINI_API_KEY`` is (see
    ``gaia_agent.tools.media.optional_media_tools``), and
    ``run_python_file`` only when ``GAIA_ALLOW_RUN_PYTHON=1`` is set (see
    ``gaia_agent.config.run_python_enabled``): it starts a program with the privileges
    of the user, so it is off until someone has read the attachment. ``deep_search`` comes last and only when
    ``GROQ_API_KEY`` is available; the run calls its ``reset_question`` before each question and may point it at
    the usage log of the run with ``set_usage_log``. The tools that
    open attachments are bound to the DATA folder of ``settings``. Every call
    builds new tool instances.
    """
    tools: list[Tool] = [
        WebSearchTool(),
        ReadWebpageTool(),
        WikipediaSearchTool(),
        WikipediaPageTool(),
        ReadFileTool(settings),
    ]
    if run_python_enabled():
        tools.append(RunPythonFileTool(settings))
    tools += [YoutubeTranscriptTool(), TranscribeAudioTool(settings), *optional_media_tools(settings)]
    if get_secret("GROQ_API_KEY") is not None:
        tools.append(DeepSearchTool())
    return tools
