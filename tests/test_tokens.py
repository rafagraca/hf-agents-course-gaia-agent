"""Tests for the rough token estimate shared by the token budget and the memory trimming."""

from __future__ import annotations

from smolagents.models import ChatMessage, MessageRole

from gaia_agent.tokens import CHARS_PER_TOKEN, estimate_tokens, text_length


def test_the_estimate_is_the_text_length_over_three_and_a_half_rounded_up() -> None:
    assert CHARS_PER_TOKEN == 3.5
    assert estimate_tokens([{"role": "user", "content": "x" * 350}]) == 100
    assert estimate_tokens([{"role": "user", "content": "x" * 351}]) == 101


def test_no_messages_cost_nothing() -> None:
    assert estimate_tokens([]) == 0


def test_the_text_of_chat_messages_and_of_plain_dicts_is_counted_alike() -> None:
    parts = [{"type": "text", "text": "a" * 10}, {"type": "text", "text": "b" * 4}]

    assert text_length(ChatMessage(role=MessageRole.USER, content=parts)) == 14
    assert text_length({"role": "user", "content": parts}) == 14
    assert text_length(ChatMessage(role=MessageRole.USER, content="c" * 7)) == 7


def test_things_that_are_not_text_count_for_nothing() -> None:
    image = {"type": "image", "image": object()}

    assert text_length(ChatMessage(role=MessageRole.USER, content=[image])) == 0
    assert text_length({"role": "user", "content": None}) == 0
    assert text_length(object()) == 0
