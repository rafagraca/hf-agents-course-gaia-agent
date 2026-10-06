"""Tests for hiding the API keys and tokens in anything the agent writes down: traces, answers, error texts."""

from __future__ import annotations

import pytest
from agent_fakes import PLAIN, FakeClient, ScriptedModel, code_reply
from smolagents.memory import ActionStep
from smolagents.monitoring import Timing

from gaia_agent import config, run
from gaia_agent._secrets import MASK, MIN_SECRET_LENGTH, SECRET_NAMES, mask_secrets
from gaia_agent.agent import AnswerResult, answer_question, build_agent
from gaia_agent.api import Question
from gaia_agent.config import Settings
from gaia_agent.trace import format_trace

GROQ_KEY = "groq-key-0123456789"
HF_TOKEN = "hf-token-0123456789"
GEMINI_KEY = "gemini-key-0123456789"


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    monkeypatch.setenv("HF_TOKEN", HF_TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)


def test_every_secret_value_is_masked(keys: None) -> None:
    text = f"a {GROQ_KEY} b {HF_TOKEN} c {GEMINI_KEY} d {GROQ_KEY}"

    assert mask_secrets(text) == f"a {MASK} b {MASK} c {MASK} d {MASK}"


def test_text_without_a_secret_is_returned_as_it_is(keys: None) -> None:
    assert mask_secrets("nothing to hide\n  here ") == "nothing to hide\n  here "


def test_a_secret_stored_in_the_user_environment_is_masked_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "_persistent_env_reader", lambda name: GROQ_KEY if name == "GROQ_API_KEY" else None)

    assert mask_secrets(f"key={GROQ_KEY}") == f"key={MASK}"


def test_the_huggingface_hub_spelling_of_the_token_is_masked_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)

    assert "HUGGING_FACE_HUB_TOKEN" in SECRET_NAMES
    assert mask_secrets(HF_TOKEN) == MASK


def test_a_value_too_short_to_be_a_key_is_not_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stray ``HF_TOKEN=1`` must not turn every 1 of an answer into asterisks."""
    monkeypatch.setenv("HF_TOKEN", "1" * (MIN_SECRET_LENGTH - 1))

    assert mask_secrets("page 111111 of 11") == "page 111111 of 11"


# --- what the agent writes down ---------------------------------------------------------------------------------


def test_a_trace_never_shows_a_secret(keys: None) -> None:
    question = Question(task_id="task-1", question=f"What is {GROQ_KEY}?", level="1", file_name="")
    result = AnswerResult(
        task_id="task-1", answer=MASK, raw_answer=f"the key {GROQ_KEY}", steps=1, seconds=1.0, error=None, model="m"
    )
    step = ActionStep(
        step_number=1,
        timing=Timing(start_time=0.0, end_time=1.0),
        model_output=f"Thought: print the key {HF_TOKEN}",
        observations=f"Execution logs:\nkey {GEMINI_KEY}",
    )

    trace = format_trace(question, result, [step])

    for secret in (GROQ_KEY, HF_TOKEN, GEMINI_KEY):
        assert secret not in trace
    assert trace.count(MASK) >= 4


def test_an_answer_and_its_raw_text_never_hold_a_secret(settings: Settings, keys: None) -> None:
    model = ScriptedModel([code_reply(f"final_answer('The key is {GROQ_KEY}')")])
    agent = build_agent(settings, tools=[], model=model)

    result = answer_question(agent, PLAIN, None)

    assert GROQ_KEY not in result.answer
    assert GROQ_KEY not in result.raw_answer
    assert MASK in result.raw_answer


def test_an_error_text_never_holds_a_secret(settings: Settings, keys: None) -> None:
    model = ScriptedModel([RuntimeError(f"invalid key {GROQ_KEY} and token {HF_TOKEN}")])
    agent = build_agent(settings, tools=[], model=model)

    result = answer_question(agent, PLAIN, None)

    assert result.error is not None
    assert GROQ_KEY not in result.error and HF_TOKEN not in result.error


def test_the_run_folder_files_never_hold_a_secret(
    settings: Settings, keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient([PLAIN])
    monkeypatch.setattr(run, "ScoringClient", lambda settings: client)
    monkeypatch.setattr(run, "default_tools", lambda settings: [])
    model = ScriptedModel([code_reply(f"print('{GROQ_KEY}')"), code_reply(f"final_answer('{HF_TOKEN}')")])
    monkeypatch.setattr(run, "BudgetedModel", lambda *args, **kwargs: model)

    assert run.main([]) == 0

    written = "\n".join(path.read_text(encoding="utf-8") for path in settings.runs_dir.rglob("*") if path.is_file())
    assert written
    assert GROQ_KEY not in written and HF_TOKEN not in written
