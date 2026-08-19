# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.message_normalizer import MessageListNormalizer
from pyrit.models import Message
from pyrit.prompt_target import TargetNormalizationContext, TargetNormalizationContextState


def _make_context() -> tuple[TargetNormalizationContext, MagicMock]:
    normalizer = MagicMock(spec=MessageListNormalizer)
    normalizer.normalize_async = AsyncMock(side_effect=lambda messages: messages)
    context = TargetNormalizationContext(
        conversation_id="conversation",
        normalizers=(normalizer,),
    )
    return context, normalizer


async def test_context_normalizes_and_is_consumed_once():
    context, normalizer = _make_context()
    messages = [Message.from_prompt(prompt="request", role="user")]

    assert context.begin_normalization(conversation_id="conversation")
    assert context.state == TargetNormalizationContextState.PREPARING
    assert await context.normalize_async(messages=messages) == messages
    context.mark_consumed()

    assert context.is_consumed
    assert context.begin_normalization(conversation_id="conversation") is False
    normalizer.normalize_async.assert_awaited_once_with(messages)


def test_context_can_restore_pending_after_preparation_failure():
    context, _ = _make_context()

    assert context.begin_normalization(conversation_id="conversation")
    context.restore_pending()

    assert context.is_pending
    assert context.begin_normalization(conversation_id="conversation")


def test_context_rejects_concurrent_preparation():
    context, _ = _make_context()

    assert context.begin_normalization(conversation_id="conversation")

    with pytest.raises(RuntimeError, match="already in progress"):
        context.begin_normalization(conversation_id="conversation")


def test_context_rejects_another_conversation():
    context, _ = _make_context()

    with pytest.raises(ValueError, match="belongs to conversation"):
        context.begin_normalization(conversation_id="other")
