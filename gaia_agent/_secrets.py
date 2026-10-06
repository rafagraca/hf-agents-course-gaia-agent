"""Hide the API keys and tokens in anything that is written down or shown.

Keys are only ever read through :func:`gaia_agent.config.get_secret` and never put in a file or a log on purpose.
This is the net under that rule for the texts the agent does not control: an error that quotes the request that
failed, a model that prints what it found in a page, an observation. Every text that is saved (traces, answers)
or shown (errors) goes through :func:`mask_secrets` first.
"""

from __future__ import annotations

from gaia_agent.config import get_secret

__all__ = ["MASK", "MIN_SECRET_LENGTH", "SECRET_NAMES", "mask_secrets"]

MASK = "***"
SECRET_NAMES = ("GROQ_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GEMINI_API_KEY")
MIN_SECRET_LENGTH = 6  # a shorter value is no key: masking it would only eat ordinary text


def mask_secrets(text: str) -> str:
    """``text`` with the value of every known secret replaced by ``***``; nothing else is changed."""
    for name in SECRET_NAMES:
        secret = get_secret(name)
        if secret and len(secret) >= MIN_SECRET_LENGTH:
            text = text.replace(secret, MASK)
    return text
