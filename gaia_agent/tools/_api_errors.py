"""Turn the error reply of an HTTP API into a short message that never contains a secret."""

from __future__ import annotations

import requests

from gaia_agent.tools._local_files import redact

ERROR_TEXT_LIMIT = 200


def error_message(response: requests.Response, *secrets: str | None) -> str:
    """The API's own error text with every secret removed, on one line and shortened.

    Both Groq and Google answer errors with ``{"error": {"message": ...}}``; any
    other body is used as it is. Secrets are removed *before* shortening, so a
    key cut in half by the limit cannot survive.
    """
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = response.content.decode("utf-8", errors="replace")
    return redact(" ".join(str(message).split()), *secrets)[:ERROR_TEXT_LIMIT]
