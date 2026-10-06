"""Doubles shared by the agent and command line tests: a scripted model, a toy tool and a fake scoring client.

Nothing here touches the network. Every question, task id and answer is invented.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from smolagents import Model, Tool
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import TokenUsage

from gaia_agent.api import Question

SCRIPTED_MODEL_ID = "fake/scripted-model"

PLAIN = Question(task_id="task-plain", question="What is six times seven?", level="1", file_name="")
SHEET = Question(
    task_id="task-sheet", question="What is the total in the sheet?", level="1", file_name="task-sheet.xlsx"
)
AUDIO = Question(task_id="task-audio", question="Which word is said twice?", level="1", file_name="task-audio.mp3")


class ScriptedModel(Model):
    """Replays canned replies in order (an exception in the script is raised) and records every request."""

    def __init__(self, replies: Sequence[str | BaseException] = (), model_id: str = SCRIPTED_MODEL_ID) -> None:
        super().__init__(model_id=model_id)
        self.replies = list(replies)
        self.requests: list[list[ChatMessage]] = []

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        self.requests.append(list(messages))
        if not self.replies:
            raise AssertionError("the scripted model ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=reply,
            token_usage=TokenUsage(input_tokens=120, output_tokens=30),
        )


class EchoTool(Tool):
    """A toy tool that repeats its input, optionally padded to a given length."""

    name = "echo_tool"
    description = "Return the text it is given."
    inputs = {
        "text": {"type": "string", "description": "Any text."},
        "pad_to": {"type": "integer", "description": "Pad the reply to this length.", "nullable": True},
    }
    output_type = "string"

    def forward(self, text: str, pad_to: int = 0) -> str:
        reply = f"echo: {text}"
        return reply + "." * max(0, (pad_to or 0) - len(reply))


class AttachmentAwareTool(Tool):
    """A toy tool that, like ``run_python_file``, is told which attachment the question being answered has."""

    name = "attachment_tool"
    description = "Return the name of the attachment of the current question."
    inputs: dict[str, dict[str, Any]] = {}
    output_type = "string"

    def __init__(self) -> None:
        super().__init__()
        self.registered: list[Path | None] = []

    def set_attachment(self, path: Path | None) -> None:
        self.registered.append(path)

    def forward(self) -> str:
        current = self.registered[-1] if self.registered else None
        return f"attachment: {current.name if current else 'none'}"


class FakeClient:
    """Stands in for ``ScoringClient``: serves fixed questions and pretends to download attachments."""

    def __init__(self, questions: Sequence[Question], files: dict[str, Path] | None = None) -> None:
        self.questions = list(questions)
        self.files = dict(files or {})
        self.downloads: list[str] = []
        self.refreshes: list[bool] = []

    def get_questions(self, refresh: bool = False) -> list[Question]:
        self.refreshes.append(refresh)
        return list(self.questions)

    def download_file(self, q: Question) -> Path | None:
        self.downloads.append(q.task_id)
        return self.files.get(q.task_id)


def code_reply(code: str, thought: str = "I will work it out.") -> str:
    """A model reply in the Thought / code block format the agent's parser expects."""
    return f"Thought: {thought}\n<code>\n{code}\n</code>"


def final_reply(answer: object) -> str:
    """A model reply that ends the run with ``final_answer(answer)``."""
    return code_reply(f"final_answer({answer!r})", thought="I have the answer.")


def message_text(message: ChatMessage) -> str:
    """The text of a chat message, whatever shape its content has."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def request_text(request: list[ChatMessage]) -> str:
    """All the text sent to the model in one request."""
    return "\n".join(message_text(message) for message in request)
