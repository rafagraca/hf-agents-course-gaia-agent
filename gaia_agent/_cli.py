"""Small helpers shared by the command line entry points (``run`` and ``submit``)."""

from __future__ import annotations

import logging
import sys

EXIT_OK = 0
EXIT_FAILED = 1  # the command ran, but something it was asked to do failed
EXIT_USAGE = 2  # bad arguments or an unusable setup: nothing was done
EXIT_INTERRUPTED = 130

LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def configure_output(verbose: bool = False) -> None:
    """Send warnings (and, with ``verbose``, this package's progress) to stderr.

    Characters the console cannot encode are escaped instead of crashing the command.
    """
    logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
    logging.getLogger("gaia_agent").setLevel(logging.INFO if verbose else logging.WARNING)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="backslashreplace")


def say(text: str) -> None:
    """Print a line of the command's normal output."""
    print(text, file=sys.stdout, flush=True)


def complain(text: str) -> None:
    """Print a line on stderr: an error, or why the command stopped."""
    print(text, file=sys.stderr, flush=True)


def short_id(task_id: str) -> str:
    """The first characters of a task id, enough to tell the questions of a run apart."""
    return task_id[:8]
