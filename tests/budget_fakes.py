"""Test doubles for the token budget tests: a fake clock, scripted models, realistic provider errors
and a loopback server that answers like Groq."""

from __future__ import annotations

import dataclasses
import json
import os

# Importing litellm downloads its price list from GitHub unless told to use the bundled copy.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import httpx
import litellm
import pytest
from smolagents import Model
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import TokenUsage

from gaia_agent.budget import BudgetedModel
from gaia_agent.config import Settings

PRIMARY = "groq/openai/gpt-oss-120b"
SECOND = "groq/qwen/qwen3.8-27b"
THIRD = "groq/openai/gpt-oss-20b"
GROQ_SECRET = "groq-secret-for-tests"
GEMINI_SECRET = "gemini-secret-for-tests"

_ORG = "in organization `org_test` service tier `on_demand`"
TPM_MESSAGE = (
    f"Rate limit reached for model `openai/gpt-oss-120b` {_ORG} on tokens per minute (TPM): "
    "Limit 8000, Used 7500, Requested 900. Please try again in 3.0s."
)
TPD_MESSAGE = (
    f"Rate limit reached for model `openai/gpt-oss-120b` {_ORG} on tokens per day (TPD): "
    "Limit 200000, Used 199500, Requested 1200. Please try again in 6m30s."
)
TOO_LARGE_MESSAGE = (
    f"Request too large for model `openai/gpt-oss-120b` {_ORG} on tokens per minute (TPM): "
    "Limit 8000, Requested 9071, please reduce your message size and try again."
)
# What Groq answers (HTTP 400) when gpt-oss replies with a tool call although no tool was offered.
BAD_GENERATION_MESSAGE = (
    'GroqException - {"error":{"message":"Tool choice is none, but model called a tool",'
    '"type":"invalid_request_error","code":"tool_use_failed",'
    '"failed_generation":"{\\"name\\": \\"code\\", \\"arguments\\": final_answer(3)}"}}'
)


class FakeClock:
    """A monotonic clock whose ``sleep`` moves time forward instead of waiting."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    @property
    def slept(self) -> float:
        return sum(self.sleeps)


def reply(text: str = "ok", tokens_in: int = 100, tokens_out: int = 50, *, usage: bool = True) -> ChatMessage:
    """A model reply, with the usage figures a real LiteLLM reply carries (unless ``usage`` is off)."""
    token_usage = TokenUsage(input_tokens=tokens_in, output_tokens=tokens_out) if usage else None
    return ChatMessage(role=MessageRole.ASSISTANT, content=text, token_usage=token_usage)


def prompt(chars: int) -> list[ChatMessage]:
    """A one-message prompt of ``chars`` characters (``chars / 3.5`` estimated tokens)."""
    return [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "x" * chars}])]


def rate_limit_error(message: str = TPM_MESSAGE, retry_after: str | None = None) -> litellm.RateLimitError:
    """The exception litellm raises for a Groq 429, with the ``retry-after`` header when given."""
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    return litellm.RateLimitError(message=message, llm_provider="groq", model="openai/gpt-oss-120b", response=response)


def bad_generation_error(message: str = BAD_GENERATION_MESSAGE) -> litellm.BadRequestError:
    """The exception litellm raises for that 400."""
    return litellm.BadRequestError(message=message, model="openai/gpt-oss-120b", llm_provider="groq")


class HttpError(Exception):
    """An error that only carries an HTTP status, like the exceptions of many SDKs."""

    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Call:
    """What a fake model was asked."""

    messages: list[ChatMessage]
    stop_sequences: list[str] | None
    response_format: dict[str, str] | None
    tools: list[Any] | None
    kwargs: dict[str, Any]


class FakeModel(Model):
    """A scripted stand-in for ``LiteLLMModel``: every call plays the next reply or raises the next error."""

    def __init__(self, model_id: str, script: list[Any], clock: FakeClock | None = None, latency: float = 0.0) -> None:
        super().__init__(model_id=model_id)
        self.calls: list[Call] = []
        self._script = list(script)
        self._clock = clock
        self._latency = latency

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        self.calls.append(Call(messages, stop_sequences, response_format, tools_to_call_from, dict(kwargs)))
        if self._clock is not None:
            self._clock.now += self._latency
        if not self._script:
            raise AssertionError(f"{self.model_id}: the script has no reply left")
        step = self._script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


class FakeFactory:
    """Builds the fake models and remembers which (model id, API key) pairs were asked for."""

    def __init__(self, models: dict[str, FakeModel]) -> None:
        self.models = models
        self.built: list[tuple[str, str | None]] = []

    def __call__(self, model_id: str, api_key: str | None) -> FakeModel:
        self.built.append((model_id, api_key))
        return self.models[model_id]


@dataclass(frozen=True)
class Rig:
    """A budgeted model wired to fake models and a fake clock."""

    model: BudgetedModel
    clock: FakeClock
    fakes: dict[str, FakeModel]
    factory: FakeFactory


def provide_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the test the provider keys the fake models are 'built' with."""
    monkeypatch.setenv("GROQ_API_KEY", GROQ_SECRET)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_SECRET)


def make_rig(
    settings: Settings,
    scripts: dict[str, list[Any]],
    *,
    model_ids: Sequence[str] | None = None,
    usage_log: Any = None,
    latency: float = 0.0,
    **overrides: Any,
) -> Rig:
    """Wire a ``BudgetedModel`` to scripted fake models (``scripts``: model id -> replies and errors)."""
    clock = FakeClock()
    fakes = {model_id: FakeModel(model_id, script, clock, latency) for model_id, script in scripts.items()}
    factory = FakeFactory(fakes)
    config = dataclasses.replace(settings, model_id=PRIMARY, fallback_model_ids=(SECOND, THIRD), **overrides)
    model = BudgetedModel(config, usage_log, model_ids, model_factory=factory, clock=clock, sleep=clock.sleep)
    return Rig(model, clock, fakes, factory)


def stub_litellm_completion(
    monkeypatch: pytest.MonkeyPatch, responder: Callable[[int, dict[str, Any]], Any]
) -> list[dict[str, Any]]:
    """Replace ``litellm.completion``; ``responder(call_number, kwargs)`` returns or raises.

    Returns the list that collects the keyword arguments of every call.
    """
    calls: list[dict[str, Any]] = []

    def completion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return responder(len(calls), kwargs)

    monkeypatch.setattr(litellm, "completion", completion)
    return calls


def completion_response(text: str = "hi", prompt_tokens: int = 11, completion_tokens: int = 7) -> SimpleNamespace:
    """The few attributes of a litellm response that ``LiteLLMModel.generate`` reads."""
    message = SimpleNamespace(role="assistant", content=text, tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


# What a loopback server replies: (HTTP status, extra headers, JSON body).
ServerReply = tuple[int, dict[str, str], dict[str, Any]]


def error_reply(status: int, message: str, retry_after: str | None = None) -> ServerReply:
    """An error answer in the shape of Groq's API."""
    body = {"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}}
    return status, ({"retry-after": retry_after} if retry_after else {}), body


def refused_generation_reply() -> ServerReply:
    """Groq's HTTP 400 for a model reply it rejected: a tool call nobody offered."""
    error = {
        "message": "Tool choice is none, but model called a tool",
        "type": "invalid_request_error",
        "code": "tool_use_failed",
        "failed_generation": '{"name": "code", "arguments": final_answer(3)}',
    }
    return 400, {}, {"error": error}


def chat_reply(text: str = "hello", prompt_tokens: int = 11, completion_tokens: int = 7) -> ServerReply:
    """A successful chat completion in the shape of Groq's API (litellm's Groq mapping needs ``service_tier``)."""
    message = {"role": "assistant", "content": text}
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "stub",
        "service_tier": "on_demand",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return 200, {}, body


class GroqStub:
    """A loopback HTTP server that answers like Groq's chat endpoint by playing a script of replies.

    Use it as a context manager; ``requests`` collects the JSON body of every request it received.
    """

    def __init__(self, replies: list[ServerReply]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._replies = list(replies)
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # the name is the http.server convention
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                stub.requests.append(body)
                status, headers, payload = stub._replies.pop(0)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                """Keep the test output quiet."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/openai/v1"

    def __enter__(self) -> GroqStub:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
