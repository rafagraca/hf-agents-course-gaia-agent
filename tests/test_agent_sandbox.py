"""What the code written by the model can and cannot do.

The model's code runs in the smolagents ``LocalPythonExecutor``, which its own documentation says is
not a security sandbox. The agent is only as safe as the set of modules it is given, so these tests pin
that set down: no module that reads or writes files, opens connections, unpickles data or starts
programs may be reachable, whatever the model (or a prompt injected into a web page) tells it to do.
The tools are the only door to the outside, and they have their own guards.
"""

from __future__ import annotations

import pytest
from agent_fakes import EchoTool, ScriptedModel
from smolagents.local_python_executor import InterpreterError, LocalPythonExecutor

from gaia_agent.agent import AUTHORIZED_IMPORTS, build_agent
from gaia_agent.config import Settings

# Everything the old import list let through (pandas and numpy read and write arbitrary files, fetch URLs and
# unpickle data), plus the usual suspects.
FORBIDDEN_IMPORTS = [
    "pandas",
    "numpy",
    "os",
    "sys",
    "io",
    "builtins",
    "subprocess",
    "socket",
    "ssl",
    "pickle",
    "shelve",
    "marshal",
    "ctypes",
    "pathlib",
    "shutil",
    "tempfile",
    "glob",
    "zipfile",
    "tarfile",
    "codecs",
    "logging",
    "importlib",
    "urllib.request",
    "http.client",
    "multiprocessing",
    "threading",
    "requests",
    "pypdf",
    "docx",
]

ESCAPES = {
    "open() for writing": "open('planted.txt', 'w')",
    "open() for reading": "open('planted.txt')",
    "__import__": "__import__('os')",
    "eval": "eval('1 + 1')",
    "exec": "exec('x = 1')",
    "compile": "compile('1', 'x', 'eval')",
    "a dunder attribute": "getattr((), '__class__')",
    "a module hidden inside random": "import random\nrandom._os.system('echo hi')",
    "a module hidden inside json": "import json\njson.codecs",
    "a module hidden inside re": "import re\nre.enum.sys",
    "a module hidden inside collections": "import collections\ncollections._sys",
}


def run_code(settings: Settings, code: str):
    """Run ``code`` in the executor of an agent built like the real one; return the executor output."""
    agent = build_agent(settings, tools=[EchoTool()], model=ScriptedModel())
    agent.python_executor.send_tools({**agent.tools})
    return agent.python_executor(code)


@pytest.mark.parametrize("module", FORBIDDEN_IMPORTS)
def test_the_model_cannot_import_a_module_that_touches_files_networks_or_processes(
    settings: Settings, module: str
) -> None:
    with pytest.raises(InterpreterError, match="is not allowed"):
        run_code(settings, f"import {module}")


@pytest.mark.parametrize("module", ["pandas", "numpy", "os", "subprocess"])
def test_the_from_form_of_an_import_is_refused_too(settings: Settings, module: str) -> None:
    with pytest.raises(InterpreterError, match="not allowed"):
        run_code(settings, f"from {module} import *")


@pytest.mark.parametrize("code", ESCAPES.values(), ids=ESCAPES.keys())
def test_the_usual_ways_out_of_the_executor_are_closed(settings: Settings, code: str) -> None:
    with pytest.raises(InterpreterError):
        run_code(settings, code)


def test_pandas_and_numpy_are_not_in_the_authorized_imports() -> None:
    assert {"pandas", "numpy"}.isdisjoint(AUTHORIZED_IMPORTS)


def test_the_agent_does_not_write_files_in_the_working_folder(
    settings: Settings, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    attempts = ["import pandas as pd\npd.Series([1]).to_csv('leak.csv')", "import numpy as np\nnp.save('leak', [1])"]

    for code in attempts:
        with pytest.raises(InterpreterError):
            run_code(settings, code)

    assert list(tmp_path.glob("leak*")) == []


def test_the_modules_that_remain_still_do_the_calculations_a_question_needs(settings: Settings) -> None:
    code = "\n".join(
        [
            "import csv, statistics, collections, re, math, datetime, json, fractions, decimal, itertools",
            "rows = list(csv.reader(['name,score', 'a,2', 'b,4', 'c,9']))",
            "scores = [int(score) for _, score in rows[1:]]",
            "counts = collections.Counter(word for word in re.findall(r'[a-z]+', 'to be or not to be'))",
            "assert counts['to'] == 2 and math.isqrt(16) == 4 and json.loads('[1]') == [1]",
            "assert datetime.date(2024, 3, 1).year == 2024",
            "assert fractions.Fraction(1, 2) + 1 == fractions.Fraction(3, 2)",
            "assert decimal.Decimal('0.1') + decimal.Decimal('0.2') == decimal.Decimal('0.3')",
            "assert list(itertools.islice(itertools.count(), 3)) == [0, 1, 2]",
            "final_answer(statistics.median(scores))",
        ]
    )

    output = run_code(settings, code)

    assert output.is_final_answer
    assert output.output == 4


def test_the_tools_stay_reachable_from_the_code(settings: Settings) -> None:
    output = run_code(settings, "final_answer(echo_tool(text='hi'))")

    assert output.output == "echo: hi"


def test_the_executor_is_a_local_one_without_wildcard_imports(settings: Settings) -> None:
    agent = build_agent(settings, tools=[], model=ScriptedModel())

    assert isinstance(agent.python_executor, LocalPythonExecutor)
    assert "*" not in agent.authorized_imports
