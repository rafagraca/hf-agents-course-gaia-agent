"""The usage log: one JSON line per model call, to check token estimates and spend after a run."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class UsageLog:
    """Appends one JSON line per model call to ``path`` (JSON Lines); does nothing when ``path`` is None.

    A line holds only the model name, token counts and times: never a prompt, a reply or a key.
    A write failure is reported with a warning and never raised, because the reply was already paid for.
    """

    def __init__(self, path: Path | str | None) -> None:
        self._path = Path(path) if path is not None else None

    def append(
        self,
        model_id: str,
        *,
        tokens: tuple[int | None, int | None],
        seconds: float,
        estimate: int,
        max_tokens: int,
        attempts: int,
        waited: float,
    ) -> None:
        """Log one call.

        ``tokens`` is the provider's real ``(input, output)`` count, ``None`` for what the reply did not
        report; ``estimate`` is our estimate of the input, to compare with the real figure; ``seconds`` is
        the time of the successful request alone and ``waited`` the time spent waiting for the minute's
        budget and between retries.
        """
        if self._path is None:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": model_id,
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "seconds": round(seconds, 3),
            "estimated_input_tokens": estimate,
            "max_tokens": max_tokens,
            "attempts": attempts,
            "waited_seconds": round(waited, 3),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            logger.warning("Could not write the usage log %s: %s", self._path, reason)
