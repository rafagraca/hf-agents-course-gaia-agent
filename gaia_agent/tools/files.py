"""File tools: read question attachments and run Python attachments. Outputs are compact and truncated.

Both tools only touch files inside the attachments folder (see
:class:`gaia_agent.tools._local_files.AttachmentSandbox`): the agent cannot use
them to read anything else on the machine, not even the rest of the DATA folder.
Expected failures are returned as a short ``Error: ...`` string so the agent can
react to them.

``settings`` fixes the DATA folder; without it the folder comes from the
environment (``GAIA_DATA_DIR``) when the tool is first used, so building a tool
never touches the disk.
"""

from __future__ import annotations

from pathlib import Path

from smolagents import Tool

from gaia_agent.config import Settings
from gaia_agent.tools._local_files import AttachmentSandbox, ToolError
from gaia_agent.tools._pyrun import DEFAULT_TIMEOUT_S, format_run_result, run_python_script
from gaia_agent.tools._readers import read_local_file


class ReadFileTool(Tool):
    """Read a question attachment (spreadsheet, text, PDF, Word, PowerPoint) as compact text."""

    name = "read_file"
    description = "Read an attachment (text, csv, xlsx, pdf, docx, pptx) as text."
    inputs = {"file_path": {"type": "string", "description": "Attachment file name or path."}}
    output_type = "string"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self._settings = settings

    def forward(self, file_path: str) -> str:
        try:
            return read_local_file(AttachmentSandbox.from_settings(self._settings).resolve_file(file_path))
        except ToolError as exc:
            return f"Error: {exc}"


class RunPythonFileTool(Tool):
    """Run the Python attachment of the current question and return its exit code, stdout and stderr.

    Meant for the ``.py`` file a question comes with ("what does this script print?"). The script runs in
    a throw-away process: a 20 s time limit, empty stdin, no credentials in its environment and a fresh
    working folder. That is good manners, not a security sandbox: the script runs with the privileges of
    the user and could read what the user can read, the Windows user environment (where tokens may be
    stored) included. Hence two guards. The tool is only offered when ``GAIA_ALLOW_RUN_PYTHON=1`` (read
    the attachment first), and it runs only the file registered with :meth:`set_attachment` as the
    attachment of the question being answered (``answer_question`` does that), never a file the agent
    wrote or found.
    """

    name = "run_python_file"
    description = "Run the question's .py attachment (20 s limit); returns exit code, stdout and stderr."
    inputs = {"file_path": {"type": "string", "description": "Attachment file name or path."}}
    output_type = "string"

    def __init__(self, settings: Settings | None = None, *, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        super().__init__()
        self._settings = settings
        self._timeout = timeout
        self._attachment: Path | None = None

    def set_attachment(self, path: Path | None) -> None:
        """Register the attachment of the question being answered (None withdraws it): the only file to run.

        The path is kept in memory only. A file in DATA cannot vouch for itself, because anything that
        can write there could plant a script and then ask for it to be run.
        """
        self._attachment = None if path is None else Path(path).resolve()

    def forward(self, file_path: str) -> str:
        try:
            sandbox = AttachmentSandbox.from_settings(self._settings)
            script = sandbox.resolve_file(file_path)
            if script.suffix.lower() != ".py":
                raise ToolError(f"only .py files can be run, not {script.suffix or 'a file without extension'}")
            self._require_attachment(script)
            result = run_python_script(script, timeout=self._timeout, scratch_dir=sandbox.scratch_dir)
            return format_run_result(result, self._timeout)
        except ToolError as exc:
            return f"Error: {exc}"

    def _require_attachment(self, script: Path) -> None:
        if self._attachment is None:
            raise ToolError("this question has no Python attachment, so there is nothing to run")
        if script != self._attachment:
            raise ToolError("only the .py attachment of the current question can be run")
