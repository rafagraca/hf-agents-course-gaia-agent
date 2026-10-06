"""Tests for ``gaia_agent.tools.files``: the ``read_file`` tool and the attachment sandbox.

Every file is generated inside the test (invented content only) and lives in the
temporary DATA folder, so nothing touches the network or the real DATA folder.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from gaia_agent.config import Settings
from gaia_agent.tools._local_files import (
    redact,
    truncate_middle,
    truncate_text,
)
from gaia_agent.tools.files import ReadFileTool

MARKER = re.compile(r"\[truncated: (\d+) chars total\]")
TEXT_LIMIT = 8000


@pytest.fixture
def attachments(settings: Settings) -> Path:
    """The folder where question attachments live (inside DATA)."""
    return settings.files_dir


@pytest.fixture
def read_file(settings: Settings) -> ReadFileTool:
    return ReadFileTool(settings)


# --------------------------------------------------------------------------- helpers


def test_truncate_text_keeps_short_text_untouched() -> None:
    assert truncate_text("short", 10) == "short"


def test_truncate_text_marks_the_total_length() -> None:
    result = truncate_text("x" * 50, 10)

    assert result.startswith("x" * 10)
    assert result.endswith("[truncated: 50 chars total]")


def test_truncate_middle_keeps_short_text_untouched() -> None:
    assert truncate_middle("short", 10) == "short"


def test_truncate_middle_keeps_both_ends_and_marks_the_cut() -> None:
    result = truncate_middle("A" * 10 + "m" * 100 + "Z" * 10, 30)

    assert result.startswith("A" * 10)
    assert result.endswith("Z" * 10)
    assert "[truncated: 120 chars total]" in result
    assert "m" * 50 not in result


def test_redact_hides_every_secret_value() -> None:
    assert redact("key=abc123 and tok=xyz", "abc123", None, "xyz") == "key=[redacted] and tok=[redacted]"


# --------------------------------------------------------------------------- read_file: text


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("script.py", "print('hello')\n"),
        ("notes.txt", "plain notes\nsecond line"),
        ("readme.md", "# Title\n\nbody"),
        ("data.json", '{"key": [1, 2, 3]}'),
        ("page.html", "<html><body><p>Hi</p></body></html>"),
        ("doc.xml", "<root><item>1</item></root>"),
        ("graph.jsonld", '{"@id": "x"}'),
    ],
)
def test_read_file_returns_text_files_verbatim(
    read_file: ReadFileTool, attachments: Path, name: str, content: str
) -> None:
    (attachments / name).write_text(content, encoding="utf-8")

    assert read_file(name) == content


def test_read_file_truncates_long_text_with_the_total_length(read_file: ReadFileTool, attachments: Path) -> None:
    (attachments / "long.txt").write_text("a" * 20_000, encoding="utf-8")

    result = read_file("long.txt")

    assert result.startswith("a" * TEXT_LIMIT)
    assert MARKER.search(result).group(1) == "20000"
    assert len(result) < TEXT_LIMIT + 100


def test_read_file_reads_latin_1_text(read_file: ReadFileTool, attachments: Path) -> None:
    (attachments / "latin.txt").write_bytes("caf\xe9 cr\xe8me".encode("latin-1"))

    assert read_file("latin.txt") == "caf\xe9 cr\xe8me"


def test_read_file_reports_an_empty_file(read_file: ReadFileTool, attachments: Path) -> None:
    (attachments / "empty.txt").write_text("", encoding="utf-8")

    assert read_file("empty.txt") == "(the file is empty)"


def test_read_file_refuses_an_unknown_binary_file(read_file: ReadFileTool, attachments: Path) -> None:
    (attachments / "blob.bin").write_bytes(b"\x00\x01\x02binary\x00")

    result = read_file("blob.bin")

    assert result.startswith("Error:")
    assert "binary" in result


def test_read_file_only_reads_the_start_of_a_huge_text_file(read_file: ReadFileTool, attachments: Path) -> None:
    (attachments / "huge.log").write_bytes(b"line\n" * 2_000_000)  # 10 MB

    result = read_file("huge.log")

    assert result.startswith("line\n")
    assert "10000000 bytes" in result
    assert len(result) < TEXT_LIMIT + 200


# --------------------------------------------------------------------------- read_file: spreadsheets


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)


def test_read_file_lists_every_sheet_with_columns_types_and_all_rows(
    read_file: ReadFileTool, attachments: Path
) -> None:
    write_workbook(
        attachments / "book.xlsx",
        {
            "Sales": pd.DataFrame({"item": ["alpha", "beta", "gamma"], "qty": [1, 2, 3], "price": [2.5, 3.0, 7.25]}),
            "Notes": pd.DataFrame({"text": ["first", "second"]}),
        },
    )

    result = read_file("book.xlsx")

    assert "2 sheets" in result
    assert "Sales" in result
    assert "Notes" in result
    assert "qty (int64)" in result
    assert "price (float64)" in result
    assert "gamma,3,7.25" in result
    assert "second" in result


def test_read_file_shows_all_rows_up_to_200(read_file: ReadFileTool, attachments: Path) -> None:
    frame = pd.DataFrame({"label": [f"row-{i:03d}" for i in range(200)], "n": range(200)})
    write_workbook(attachments / "edge.xlsx", {"Data": frame})

    result = read_file("edge.xlsx")

    assert "row-000" in result
    assert "row-199" in result
    assert not MARKER.search(result)


def test_read_file_summarises_sheets_with_more_than_200_rows(read_file: ReadFileTool, attachments: Path) -> None:
    frame = pd.DataFrame({"label": [f"row-{i:03d}" for i in range(250)], "n": range(250)})
    write_workbook(attachments / "big.xlsx", {"Data": frame})

    result = read_file("big.xlsx")

    assert "250 rows" in result
    assert "first 50" in result
    assert "row-049" in result
    assert "row-050" not in result
    assert "describe()" in result
    assert "mean" in result


def test_read_file_handles_an_empty_sheet(read_file: ReadFileTool, attachments: Path) -> None:
    write_workbook(attachments / "blank.xlsx", {"Empty": pd.DataFrame({"only": []})})

    result = read_file("blank.xlsx")

    assert "Empty" in result
    assert "0 rows" in result


def test_read_file_handles_a_sheet_without_columns(read_file: ReadFileTool, attachments: Path) -> None:
    write_workbook(attachments / "void.xlsx", {"Void": pd.DataFrame()})

    result = read_file("void.xlsx")

    assert "Void" in result
    assert "no columns" in result


def test_read_file_reads_comma_separated_csv(read_file: ReadFileTool, attachments: Path) -> None:
    (attachments / "table.csv").write_text("name,score\nann,3\nbob,5\n", encoding="utf-8")

    result = read_file("table.csv")

    assert "score (int64)" in result
    assert "bob,5" in result


def test_read_file_detects_semicolon_separated_csv(read_file: ReadFileTool, attachments: Path) -> None:
    (attachments / "table.csv").write_text("name;score\nann;3\nbob;5\n", encoding="utf-8")

    result = read_file("table.csv")

    assert "score (int64)" in result
    assert "bob,5" in result


def test_read_file_reads_a_single_column_csv(read_file: ReadFileTool, attachments: Path) -> None:
    (attachments / "one.csv").write_text("word\nalpha\nbeta\n", encoding="utf-8")

    assert "beta" in read_file("one.csv")


def test_read_file_reports_an_empty_csv(read_file: ReadFileTool, attachments: Path) -> None:
    (attachments / "none.csv").write_text("", encoding="utf-8")

    assert read_file("none.csv").startswith("Error:")


def test_read_file_reports_a_corrupt_workbook(read_file: ReadFileTool, attachments: Path) -> None:
    (attachments / "broken.xlsx").write_bytes(b"this is not a zip archive")

    result = read_file("broken.xlsx")

    assert result.startswith("Error:")
    assert "broken.xlsx" in result


def test_read_file_explains_that_a_legacy_xls_cannot_be_read(read_file: ReadFileTool, attachments: Path) -> None:
    # The `xlrd` reader is not a project dependency, so the agent is told so instead of getting a pandas traceback.
    (attachments / "old.xls").write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64)

    result = read_file("old.xls")

    assert result.startswith("Error:")
    assert "old.xls" in result
    assert "only .xlsx" in result


@pytest.mark.parametrize("name", ["huge.csv", "huge.xlsx", "huge.pdf", "huge.docx", "huge.pptx"])
def test_read_file_refuses_to_parse_a_huge_document(read_file: ReadFileTool, attachments: Path, name: str) -> None:
    with (attachments / name).open("wb") as handle:
        handle.seek(30 * 1024 * 1024)
        handle.write(b"\x00")

    result = read_file(name)

    assert result.startswith("Error:")
    assert "too large" in result
    assert "30 MB" in result


def test_read_file_caps_the_size_of_a_wide_workbook(read_file: ReadFileTool, attachments: Path) -> None:
    frame = pd.DataFrame({f"col{i}": [f"value-{i}-{j}" * 3 for j in range(150)] for i in range(30)})
    write_workbook(attachments / "wide.xlsx", {"Wide": frame})

    result = read_file("wide.xlsx")

    assert MARKER.search(result)
    assert len(result) < TEXT_LIMIT + 100
