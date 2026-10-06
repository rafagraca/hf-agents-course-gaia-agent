"""The repository has no way to compare answers with the GAIA reference answers, and keeps the guards against it.

* no module scores answers (``evaluate`` and ``scoring`` are gone) and no dependency exists only to read the
  benchmark metadata (``pyarrow``);
* the only code that touches the Hugging Face Hub is ``api.py``, and it downloads one thing: the attachment a
  question names;
* an answer key found anywhere in the data folder stops a run.
"""

from __future__ import annotations

import ast
import importlib.util
import tomllib
from pathlib import Path

import pytest
from api_fakes import (
    FakeHub,
    client,  # noqa: F401  (fixtures)
    hf_token,  # noqa: F401
    hub,  # noqa: F401
    rsps,  # noqa: F401
    sleeps,  # noqa: F401
)
from test_api_download import named

from gaia_agent import config, run
from gaia_agent.api import ScoringClient, UnsafeFileNameError, sanitize_file_name
from gaia_agent.config import REPO_ROOT, ConfigError, Settings

PACKAGE = REPO_ROOT / "gaia_agent"
# What reads or lists a dataset repository: only the attachment download in api.py may use ``hf_hub_download``.
HUB_LISTING_NAMES = {"snapshot_download", "list_repo_files", "list_repo_tree", "HfApi", "load_dataset", "HfFileSystem"}
FORBIDDEN_MODULES = {"pyarrow", "datasets", "fastparquet", "polars"}
RESERVED_NAMES = ["metadata.jsonl", "METADATA.PARQUET", "metadata", "metadata.level1.json", "x.parquet", "Y.JSONL"]


def package_modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def imported_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("module", ["gaia_agent.evaluate", "gaia_agent.scoring"])
def test_there_is_no_module_that_scores_answers(module: str) -> None:
    assert importlib.util.find_spec(module) is None


def test_the_package_has_no_scorer_files() -> None:
    assert not (PACKAGE / "evaluate.py").exists()
    assert not (PACKAGE / "scoring.py").exists()


def test_the_run_files_have_no_evaluation_file() -> None:
    from gaia_agent import runs

    assert not hasattr(runs, "EVALUATION_FILE")


def test_no_dependency_exists_to_read_benchmark_metadata() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        spec.split("[")[0].split(">")[0].split("<")[0].split("=")[0].strip().lower()
        for spec in project["project"]["dependencies"]
    }
    assert declared.isdisjoint(FORBIDDEN_MODULES)


def test_no_module_imports_a_reader_of_the_benchmark_metadata() -> None:
    for path in package_modules():
        assert imported_names(path).isdisjoint(FORBIDDEN_MODULES), path.name


def test_only_the_scoring_client_touches_the_hub_and_it_never_lists_or_snapshots_a_repository() -> None:
    for path in package_modules():
        names = imported_names(path)
        assert names.isdisjoint(HUB_LISTING_NAMES), path.name
        if path.name != "api.py":
            assert "huggingface_hub" not in names and "hf_hub_download" not in names, path.name


def test_the_hub_download_is_asked_for_one_file_named_after_the_question() -> None:
    source = (PACKAGE / "api.py").read_text(encoding="utf-8")
    calls = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "hf_hub_download"
    ]
    assert len(calls) == 1
    filename = next(kw.value for kw in calls[0].keywords if kw.arg == "filename")
    assert "name" in ast.unparse(filename)  # the sanitised file_name of the question, nothing else


@pytest.mark.parametrize("name", RESERVED_NAMES)
def test_names_of_metadata_or_data_files_are_never_an_attachment(name: str) -> None:
    with pytest.raises(UnsafeFileNameError):
        sanitize_file_name(name)


@pytest.mark.parametrize("name", RESERVED_NAMES)
def test_such_a_name_never_reaches_the_api_or_the_hub(
    client: ScoringClient, hub: FakeHub, hf_token: str, name: str
) -> None:  # noqa: F811
    assert client.download_file(named(name)) is None
    assert hub.calls == []


@pytest.mark.parametrize("name", ["answers.xlsx", "photo.png", "notes.txt", "script.py", "metadata_notes.txt"])
def test_ordinary_attachment_names_are_still_fine(name: str) -> None:
    assert sanitize_file_name(name) == name


def make_key(settings: Settings, relative: str) -> Path:
    path = settings.data_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "relative",
    [
        "gaia/2023/validation/metadata.jsonl",
        "cache/metadata.jsonl",
        "cache/files/Metadata.parquet",
        "runs/20260101-000000/metadata.level1.parquet",
        "tmp/deep/er/validation.parquet",
        "somewhere/metadata",
    ],
)
def test_an_answer_key_anywhere_in_the_data_folder_stops_a_run(settings: Settings, relative: str) -> None:
    key = make_key(settings, relative)

    assert key in config.answer_key_files(settings)
    with pytest.raises(ConfigError, match="answer key"):
        config.ensure_no_answer_key(settings)


def test_the_files_of_a_run_are_not_an_answer_key(settings: Settings) -> None:
    make_key(settings, "runs/20260101-000000/answers.jsonl")
    make_key(settings, "runs/20260101-000000/usage.jsonl")
    make_key(settings, "cache/questions.json")
    make_key(settings, "cache/files/photo.png")

    assert config.answer_key_files(settings) == []
    config.ensure_no_answer_key(settings)


def test_the_error_names_where_the_key_is(settings: Settings) -> None:
    key = make_key(settings, "cache/files/metadata.parquet")

    with pytest.raises(ConfigError) as caught:
        config.ensure_no_answer_key(settings)

    assert str(key) in str(caught.value)


def test_run_still_checks_for_a_key_before_it_starts() -> None:
    assert "ensure_no_answer_key" in (PACKAGE / "run.py").read_text(encoding="utf-8")
    assert run.ensure_no_answer_key is config.ensure_no_answer_key


# The tests above look at imports and at plain calls; this one reads the text, which also catches
# ``import huggingface_hub; huggingface_hub.snapshot_download(...)`` and a request to a metadata URL.
BENCHMARK_FILE_MARKERS = (
    "snapshot_download",
    "list_repo_files",
    "list_repo_tree",
    "load_dataset",
    "metadata.jsonl",
    ".parquet",
    "datasets/",
)
# These three name those files in order to refuse them (the guard of the data folder, the attachments, the web).
MODULES_THAT_REFUSE_THEM = {"config.py", "_attachments.py", "_sources.py"}


def test_no_module_other_than_the_guards_so_much_as_names_the_benchmark_files() -> None:
    offenders = [
        f"{path.name}: {marker}"
        for path in package_modules()
        if path.name not in MODULES_THAT_REFUSE_THEM
        for marker in BENCHMARK_FILE_MARKERS
        if marker in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_the_text_check_would_catch_a_module_that_lists_the_dataset(tmp_path: Path) -> None:
    sneaky = tmp_path / "sneaky.py"
    sneaky.write_text("import huggingface_hub\nhuggingface_hub.snapshot_download('x')\n", encoding="utf-8")

    assert any(marker in sneaky.read_text(encoding="utf-8") for marker in BENCHMARK_FILE_MARKERS)
