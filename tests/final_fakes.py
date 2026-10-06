"""Doubles for the tests of final runs: a scripted ``git`` and the invented commits it reports."""

from __future__ import annotations

from collections.abc import Sequence

from gaia_agent._final import GitError

COMMIT = "0123456789abcdef0123456789abcdef01234567"
OTHER_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
TAG = "v1.0.0"


class FakeGit:
    """Answers the few ``git`` commands a final run asks, from its attributes; records every call."""

    def __init__(
        self,
        *,
        status: str = "",
        commit: str = COMMIT,
        tag: str | None = TAG,
        ignored: str = "",
        tag_commit: str | None = None,
    ) -> None:
        """``status`` answers ``status --porcelain`` and ``ignored`` the same command with ``--ignored``;
        ``tag_commit`` is the commit the tag names now (default: ``commit``; ``""``: the tag does not exist)."""
        self.status, self.commit, self.tag, self.ignored = status, commit, tag, ignored
        self.tag_commit = commit if tag_commit is None else tag_commit
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args: Sequence[str]) -> str:
        self.calls.append(tuple(args))
        if args[0] == "status":
            return self.ignored if "--ignored" in args else self.status
        if args[0] == "rev-parse" and any(arg.endswith("^{commit}") for arg in args):
            if not self.tag_commit:
                raise GitError("exit code 1")
            return self.tag_commit + "\n"
        if args[0] == "rev-parse":
            return self.commit + "\n"
        if args[0] == "describe":
            if self.tag is None:
                raise GitError("fatal: no tag exactly matches")
            return self.tag + "\n"
        raise AssertionError(f"unexpected git command: {args}")
