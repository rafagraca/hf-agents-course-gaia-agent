"""The context an agent sends to the model stays inside the per-request budget, whatever the model does wrong.

Two failures seen with the real agent: an exception whose text holds a huge value (a ``KeyError`` with a 30,000
character key) and a model that answers twice in a row with long text and no code (the parsing error quotes the
reply, which the prompt already holds as the assistant turn). smolagents sends ``step.error`` in full with every
later request, so either one brought the next prompt over the per-minute token budget and ended the question.
No network: the model is scripted.
"""

from __future__ import annotations

import dataclasses

from agent_fakes import PLAIN, ScriptedModel, code_reply, final_reply

from gaia_agent.agent import answer_question, build_agent
from gaia_agent.budget import TokenBudget
from gaia_agent.config import Settings

CAPACITY = TokenBudget(8000).capacity  # what a prompt may be at most: 7,600 estimated tokens


def estimated(request: list) -> int:
    return TokenBudget(8000).estimate_tokens(request)


def test_an_exception_with_a_huge_message_does_not_bring_the_next_prompt_over_the_budget(settings: Settings) -> None:
    model = ScriptedModel([code_reply("{}['K' * 30000]"), final_reply("recovered")])
    agent = build_agent(settings, tools=[], model=model)

    result = answer_question(agent, PLAIN, None)

    assert (result.answer, result.error) == ("recovered", None)
    assert estimated(model.requests[1]) < 3000  # it was about 17,800 before the error was cut


def test_two_long_replies_without_code_in_a_row_do_not_end_the_question(settings: Settings) -> None:
    long_reply = "I am thinking about the answer, but I forgot the code. " * 110  # about 6,100 characters
    # A reply without code is asked for once more (a fresh sample costs a call, a parsing error a step), so each
    # failed step uses two replies.
    model = ScriptedModel([long_reply] * 4 + [final_reply("recovered")])
    agent = build_agent(settings, tools=[], model=model)

    result = answer_question(agent, PLAIN, None)

    assert (result.answer, result.error) == ("recovered", None)
    assert max(estimated(request) for request in model.requests) <= settings.max_prompt_tokens


def test_the_limit_of_the_settings_is_the_limit_the_agent_works_to(settings: Settings) -> None:
    tight = dataclasses.replace(settings, max_prompt_tokens=2500)
    chatty = "x = 'filler ' * 400\nprint(x)"
    model = ScriptedModel([code_reply(chatty)] * 6 + [final_reply("done")])
    agent = build_agent(tight, tools=[], model=model)

    result = answer_question(agent, PLAIN, None)

    assert (result.answer, result.error) == ("done", None)
    # the system prompt and the task alone take about 1,800 tokens, so the history was squeezed to fit
    assert max(estimated(request) for request in model.requests[1:]) <= tight.max_prompt_tokens
    assert max(estimated(request) for request in model.requests) < CAPACITY
