"""Readers that turn a local attachment into compact text for the agent.

Spreadsheets go through pandas, PDFs through pypdf, Word files through
python-docx and PowerPoint decks through the standard library (a ``.pptx`` is
a zip of XML parts). Plain text is read directly. Images, audio and video cannot
be read as text: the reader says which tool to use instead. Every result is
capped at :data:`TEXT_LIMIT` characters and ends with ``[truncated: N chars total]``
when it was cut.
"""

from __future__ import annotations

import csv
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
from docx import Document
from docx.opc.exceptions import OpcError
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader
from pypdf.errors import DependencyError, PyPdfError

from gaia_agent.config import get_secret
from gaia_agent.tools._local_files import ToolError, truncate_text

TEXT_LIMIT = 8000
FULL_TABLE_ROWS = 200
PREVIEW_ROWS = 50
MEGABYTE = 1024 * 1024
MAX_PARSE_BYTES = 30 * MEGABYTE
MAX_UNPACKED_BYTES = 150 * MEGABYTE  # what the parts of a zip-based Office file may add up to once unpacked
MAX_FULL_READ_BYTES = 5_000_000
MAX_PDF_PAGES = 60
MAX_SLIDE_XML_BYTES = 5_000_000
SNIFF_BYTES = 4096

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"})
AUDIO_SUFFIXES = frozenset(
    {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".oga", ".opus", ".aac", ".wma", ".mpga", ".amr", ".aif", ".aiff"}
)
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".m4v", ".mpeg", ".mpg", ".flv", ".3gp"})
CSV_SUFFIXES = frozenset({".csv", ".tsv"})
EXCEL_SUFFIXES = frozenset({".xlsx", ".xlsm"})  # not .xls: reading it needs xlrd, which is not a dependency
ZIP_SUFFIXES = frozenset({".docx", ".pptx", *EXCEL_SUFFIXES})  # Office files are zip archives of XML parts

# Everything that can go wrong while parsing a user-supplied document with these libraries.
READ_ERRORS = (
    OSError,
    ValueError,
    ImportError,
    KeyError,
    csv.Error,
    zipfile.BadZipFile,
    PyPdfError,
    DependencyError,  # not a PyPdfError: raised for AES-encrypted PDFs when `cryptography` is missing
    OpcError,
    ElementTree.ParseError,
)

_SLIDE_PART = re.compile(r"ppt/slides/slide(\d+)\.xml")
_DRAWING_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def read_local_file(path: Path) -> str:
    """Return the content of ``path`` as compact text, or raise :class:`ToolError`."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return _image_hint(path)
    if suffix in AUDIO_SUFFIXES:
        return f'{path.name} is an audio file. Call transcribe_audio(file_path="{path.name}") to get its transcript.'
    if suffix in VIDEO_SUFFIXES:
        return (
            f"{path.name} is a video file. Its soundtrack can be transcribed with "
            f'transcribe_audio(file_path="{path.name}"); its pictures cannot be inspected.'
        )
    if suffix == ".xls":
        raise ToolError(f"{path.name} is an old-format .xls workbook, which cannot be read here (only .xlsx can)")
    reader = _reader_for(suffix)
    try:
        if reader is not _read_text:
            _check_parse_size(path)
        return reader(path)
    except READ_ERRORS as exc:
        raise ToolError(f"could not read {path.name}: {_short_reason(exc)}") from exc


def _check_parse_size(path: Path) -> None:
    """Spreadsheets, PDFs and Office files are parsed in memory: refuse the ones that would take forever."""
    size = path.stat().st_size
    if size > MAX_PARSE_BYTES:
        raise ToolError(f"{path.name} is too large to read ({size / MEGABYTE:.0f} MB; the limit is 30 MB)")
    if path.suffix.lower() in ZIP_SUFFIXES:
        _check_unpacked_size(path)


def _check_unpacked_size(path: Path) -> None:
    """Refuse an Office file whose parts unpack to far more than it weighs (a zip bomb is tiny on disk).

    The sizes the archive declares are added up; the readers cannot take out more than a part declares.
    A file that is not a zip at all is left for the reader to report.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            unpacked = sum(info.file_size for info in archive.infolist())
    except (zipfile.BadZipFile, OSError):
        return
    if unpacked > MAX_UNPACKED_BYTES:
        limit_mb = MAX_UNPACKED_BYTES // MEGABYTE
        raise ToolError(f"{path.name} would unpack to {unpacked / MEGABYTE:.0f} MB; the limit is {limit_mb} MB")


def _reader_for(suffix: str) -> Callable[[Path], str]:
    if suffix in CSV_SUFFIXES or suffix in EXCEL_SUFFIXES:
        return _read_spreadsheet
    return {".pdf": _read_pdf, ".docx": _read_docx, ".pptx": _read_pptx}.get(suffix, _read_text)


def _short_reason(exc: Exception) -> str:
    """One line describing ``exc``: its type and the start of its message."""
    message = " ".join(str(exc).split())[:150]
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _image_hint(path: Path) -> str:
    if get_secret("GROQ_API_KEY"):
        return (
            f"{path.name} is an image, which read_file cannot show. "
            f'Call describe_image(file_path="{path.name}", question="...") to ask what it contains.'
        )
    return (
        f"{path.name} is an image, and no image-understanding tool is available in this run, "
        "so its content cannot be inspected. Do not guess what it shows."
    )


# --------------------------------------------------------------------------- plain text


def _decode(data: bytes, *, complete: bool) -> str:
    """Decode UTF-8 (with or without BOM), falling back to Windows-1252 for legacy files."""
    if not complete:  # the read stopped mid-file: a cut character must not flip the encoding
        return data.decode("utf-8-sig", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _read_text(path: Path) -> str:
    size = path.stat().st_size
    if size == 0:
        return "(the file is empty)"
    with path.open("rb") as handle:
        data = handle.read(MAX_FULL_READ_BYTES)
    if b"\x00" in data[:SNIFF_BYTES]:
        raise ToolError(f"{path.name} is a binary file of an unsupported type ({path.suffix or 'no extension'})")
    complete = size <= MAX_FULL_READ_BYTES
    text = _decode(data, complete=complete).replace("\r\n", "\n").replace("\r", "\n")
    if not complete:
        return f"{text[:TEXT_LIMIT].rstrip()}\n[truncated: file has {size} bytes in total]"
    return truncate_text(text, TEXT_LIMIT)


# --------------------------------------------------------------------------- spreadsheets


def _read_delimited(path: Path) -> pd.DataFrame:
    options = {"encoding": "utf-8-sig", "encoding_errors": "replace"}
    try:
        return pd.read_csv(path, sep=None, engine="python", **options)
    except csv.Error:  # the delimiter sniffer gives up on one-column files
        return pd.read_csv(path, **options)


def _render_table(title: str, frame: pd.DataFrame) -> str:
    rows, columns = frame.shape
    lines = [f"== {title} ({rows} rows x {columns} columns) =="]
    if columns == 0:
        return "\n".join([*lines, "(no columns)"])
    lines.append("Columns: " + ", ".join(f"{name} ({dtype})" for name, dtype in frame.dtypes.items()))
    if rows == 0:
        return "\n".join([*lines, "(no data rows)"])
    if rows <= FULL_TABLE_ROWS:
        return "\n".join([*lines, _to_csv(frame)])
    lines.append(f"Showing the first {PREVIEW_ROWS} of {rows} rows:")
    lines.append(_to_csv(frame.head(PREVIEW_ROWS)))
    lines.append("describe():")
    lines.append(frame.describe().to_string())
    return "\n".join(lines)


def _to_csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n").rstrip("\n")


def _read_spreadsheet(path: Path) -> str:
    if path.suffix.lower() in CSV_SUFFIXES:
        return truncate_text(_render_table("CSV table", _read_delimited(path)), TEXT_LIMIT)
    sheets = pd.read_excel(path, sheet_name=None)
    overview = ", ".join(f"{name} ({frame.shape[0]} rows x {frame.shape[1]} columns)" for name, frame in sheets.items())
    noun = "sheet" if len(sheets) == 1 else "sheets"
    parts = [f"{path.name}: workbook with {len(sheets)} {noun}: {overview}"]
    parts.extend(_render_table(f'Sheet "{name}"', frame) for name, frame in sheets.items())
    return truncate_text("\n\n".join(parts), TEXT_LIMIT)


# --------------------------------------------------------------------------- documents


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ToolError(f"{path.name} is password protected")
    total = len(reader.pages)
    chunks = []
    for number, page in enumerate(reader.pages[:MAX_PDF_PAGES], start=1):
        text = (page.extract_text() or "").strip()
        if text:
            chunks.append(f"--- page {number} ---\n{text}")
    if not chunks:
        return f"No extractable text in {path.name} ({total} pages): it may be a scan, so its content cannot be read."
    if total > MAX_PDF_PAGES:
        chunks.append(f"[only the first {MAX_PDF_PAGES} of {total} pages were read]")
    return truncate_text("\n".join(chunks), TEXT_LIMIT)


def _read_docx(path: Path) -> str:
    lines: list[str] = []
    table_count = 0
    for block in Document(str(path)).iter_inner_content():
        if isinstance(block, Paragraph):
            if block.text.strip():
                lines.append(block.text.strip())
        elif isinstance(block, Table):
            table_count += 1
            lines.append(f"[table {table_count}]")
            lines.extend(" | ".join(cell.text.strip() for cell in row.cells) for row in block.rows)
    if not lines:
        return "(the document has no text)"
    return truncate_text("\n".join(lines), TEXT_LIMIT)


def _slide_text(xml_bytes: bytes) -> str:
    root = ElementTree.fromstring(xml_bytes)
    paragraphs = (
        "".join(run.text or "" for run in para.iter(f"{_DRAWING_NS}t")) for para in root.iter(f"{_DRAWING_NS}p")
    )
    return " | ".join(text.strip() for text in paragraphs if text.strip())


def _read_pptx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        numbered = sorted(
            (
                (int(match.group(1)), info)
                for info in archive.infolist()
                if (match := _SLIDE_PART.fullmatch(info.filename)) and info.file_size <= MAX_SLIDE_XML_BYTES
            ),
            key=lambda item: item[0],
        )
        slides = [f"Slide {number}: {_slide_text(archive.read(info)) or '(no text)'}" for number, info in numbered]
    if not slides:
        return "(the presentation has no slides)"
    return truncate_text("\n".join(slides), TEXT_LIMIT)
