"""Tests for telling provider failures apart: rate limits, oversized requests, daily quotas, server hiccups."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
import requests
from budget_fakes import (
    BAD_GENERATION_MESSAGE,
    TOO_LARGE_MESSAGE,
    TPD_MESSAGE,
    TPM_MESSAGE,
    HttpError,
    bad_generation_error,
    litellm,
    rate_limit_error,
)

from gaia_agent.provider_errors import FailureKind, classify_failure, limit_and_requested, retry_after_seconds

RATE_LIMITED = FailureKind.RATE_LIMITED
DAILY_QUOTA = FailureKind.DAILY_QUOTA
TOO_LARGE = FailureKind.TOO_LARGE
TRANSIENT = FailureKind.TRANSIENT
BAD_GENERATION = FailureKind.BAD_GENERATION
MODEL_UNAVAILABLE = FailureKind.MODEL_UNAVAILABLE
AUTH = FailureKind.AUTH
OTHER = FailureKind.OTHER


class ResponseError(Exception):
    """An error that carries a response object of any shape."""

    def __init__(self, response: object) -> None:
        super().__init__("no hint here")
        self.response = response


def litellm_error(name: str) -> BaseException:
    """A real litellm exception, built the way litellm builds it."""
    return getattr(litellm, name)(message="provider said no", llm_provider="groq", model="m")


CLASSIFICATIONS = [
    pytest.param(rate_limit_error(TPM_MESSAGE), RATE_LIMITED, id="litellm 429 per minute"),
    pytest.param(rate_limit_error(TPD_MESSAGE), DAILY_QUOTA, id="litellm 429 per day"),
    # litellm reports a Groq 413 as a RateLimitError with status 429: the text has to win over the status.
    pytest.param(rate_limit_error(TOO_LARGE_MESSAGE), TOO_LARGE, id="litellm 413 reported as 429"),
    pytest.param(HttpError(413, "payload too big"), TOO_LARGE, id="plain 413 status"),
    pytest.param(RuntimeError("Request too large for model x"), TOO_LARGE, id="request too large, no status"),
    pytest.param(
        litellm.ContextWindowExceededError(message="too long", model="m", llm_provider="groq"),
        TOO_LARGE,
        id="context window exceeded",
    ),
    pytest.param(HttpError(429, "slow down"), RATE_LIMITED, id="plain 429 status"),
    pytest.param(RuntimeError("Rate limit exceeded"), RATE_LIMITED, id="rate limit text, no status"),
    pytest.param(RuntimeError("Error code: 429"), RATE_LIMITED, id="429 in the text, no status"),
    pytest.param(HttpError(429, "on requests per day (RPD): Limit 1000"), DAILY_QUOTA, id="requests per day"),
    pytest.param(HttpError(429, "RPD limit reached"), DAILY_QUOTA, id="RPD acronym only"),
    pytest.param(HttpError(429, "daily limit reached"), DAILY_QUOTA, id="daily word"),
    pytest.param(HttpError(429, "GenerateRequestsPerDayPerModel"), DAILY_QUOTA, id="per-day quota id"),
    pytest.param(HttpError(429, "insufficient_quota: no credit"), DAILY_QUOTA, id="insufficient quota"),
    pytest.param(HttpError(503, "service unavailable"), TRANSIENT, id="503 status"),
    pytest.param(HttpError(500, "rate limit bug"), TRANSIENT, id="a server error that mentions rate limits"),
    pytest.param(HttpError(408, "request timeout"), TRANSIENT, id="408 status"),
    pytest.param(litellm_error("InternalServerError"), TRANSIENT, id="litellm internal server error"),
    pytest.param(litellm_error("ServiceUnavailableError"), TRANSIENT, id="litellm service unavailable"),
    pytest.param(litellm_error("BadGatewayError"), TRANSIENT, id="litellm bad gateway"),
    pytest.param(litellm_error("Timeout"), TRANSIENT, id="litellm timeout"),
    pytest.param(litellm_error("APIConnectionError"), TRANSIENT, id="litellm connection error"),
    pytest.param(requests.exceptions.ReadTimeout("slow"), TRANSIENT, id="requests timeout"),
    pytest.param(requests.exceptions.ConnectionError("down"), TRANSIENT, id="requests connection error"),
    pytest.param(TimeoutError("timed out"), TRANSIENT, id="built-in timeout"),
    pytest.param(ConnectionResetError("reset"), TRANSIENT, id="built-in connection error"),
    pytest.param(bad_generation_error(), BAD_GENERATION, id="refused generation (tool_use_failed)"),
    pytest.param(
        bad_generation_error(BAD_GENERATION_MESSAGE + " Request too large, rate limit, daily"),
        BAD_GENERATION,
        id="a refused generation that quotes other error words",
    ),
    pytest.param(HttpError(404, "model not found"), MODEL_UNAVAILABLE, id="404 status"),
    pytest.param(litellm_error("NotFoundError"), MODEL_UNAVAILABLE, id="litellm not found"),
    pytest.param(
        litellm.BadRequestError(
            message='GroqException - {"error":{"message":"The model `x` has been decommissioned and is no longer '
            'supported.","type":"invalid_request_error","code":"model_decommissioned"}}',
            model="x",
            llm_provider="groq",
        ),
        MODEL_UNAVAILABLE,
        id="a decommissioned model (HTTP 400)",
    ),
    pytest.param(
        RuntimeError("The model `qwen/qwen9` does not exist or you do not have access to it."),
        MODEL_UNAVAILABLE,
        id="a model that does not exist, no status",
    ),
    pytest.param(RuntimeError('{"code": "model_not_found"}'), MODEL_UNAVAILABLE, id="model_not_found code"),
    pytest.param(HttpError(401, "invalid api key"), AUTH, id="401 status"),
    pytest.param(HttpError(403, "forbidden"), AUTH, id="403 status"),
    pytest.param(litellm_error("AuthenticationError"), AUTH, id="litellm authentication error"),
    pytest.param(
        litellm.PermissionDeniedError(
            message="blocked",
            llm_provider="groq",
            model="m",
            response=httpx.Response(403, request=httpx.Request("POST", "https://api.groq.com/")),
        ),
        AUTH,
        id="litellm permission denied",
    ),
    pytest.param(RuntimeError("used 14290 tokens"), OTHER, id="a number that contains 429"),
    pytest.param(HttpError(400, "daily report"), OTHER, id="a daily word on a bad request"),
    pytest.param(HttpError(400, "the table does not exist"), OTHER, id="400 about something else that is missing"),
    pytest.param(HttpError(429, "the model does not exist, said the rate limiter"), RATE_LIMITED, id="429 wins"),
    pytest.param(litellm_error("BadRequestError"), OTHER, id="bad request"),
    pytest.param(ValueError("boom"), OTHER, id="any other error"),
]


@pytest.mark.parametrize(("error", "expected"), CLASSIFICATIONS)
def test_failures_are_classified(error: BaseException, expected: FailureKind) -> None:
    assert classify_failure(error) is expected


RETRY_AFTERS = [
    pytest.param(rate_limit_error("no hint", retry_after="7"), 7.0, id="header in seconds"),
    pytest.param(rate_limit_error("no hint", retry_after="2.5"), 2.5, id="fractional header"),
    pytest.param(rate_limit_error(TPM_MESSAGE), 3.0, id="message in seconds"),
    pytest.param(rate_limit_error("Please try again in 1m26.4s."), 86.4, id="message in minutes and seconds"),
    pytest.param(rate_limit_error(TPD_MESSAGE), 390.0, id="message in minutes"),
    pytest.param(rate_limit_error("Please try again in 2h3m1.5s."), 7381.5, id="message in hours"),
    pytest.param(rate_limit_error("Please try again in 850ms."), 0.85, id="message in milliseconds"),
    pytest.param(rate_limit_error("try again in 7.5s", retry_after="7"), 7.5, id="message longer than header"),
    pytest.param(rate_limit_error("try again in 1s", retry_after="5"), 5.0, id="header longer than message"),
    pytest.param(rate_limit_error("Rate limit reached"), None, id="no hint at all"),
    pytest.param(rate_limit_error("nope", retry_after="soon"), None, id="unreadable header"),
    pytest.param(rate_limit_error("nope", retry_after="-3"), None, id="negative header"),
    pytest.param(ValueError("boom"), None, id="plain error"),
    pytest.param(ResponseError(SimpleNamespace(headers=None)), None, id="response without headers"),
    pytest.param(ResponseError(SimpleNamespace(headers=["retry-after"])), None, id="headers not a mapping"),
]


@pytest.mark.parametrize(("error", "expected"), RETRY_AFTERS)
def test_retry_after_comes_from_the_header_or_the_message(error: BaseException, expected: float | None) -> None:
    seconds = retry_after_seconds(error)

    if expected is None:
        assert seconds is None
    else:
        assert seconds == pytest.approx(expected)


def test_limit_and_requested_are_read_from_the_provider_message() -> None:
    assert limit_and_requested(rate_limit_error(TOO_LARGE_MESSAGE)) == (8000, 9071)
    assert limit_and_requested(rate_limit_error(TPM_MESSAGE)) == (8000, 900)


def test_limit_and_requested_are_none_without_figures() -> None:
    assert limit_and_requested(ValueError("boom")) is None
