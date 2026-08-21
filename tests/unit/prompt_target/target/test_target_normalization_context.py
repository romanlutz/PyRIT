# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid

import pytest

from pyrit.models import ChatMessageRole, Message, PromptResponseError
from pyrit.prompt_target.common.target_normalization_context import (
    TargetNormalizationContext,
    filter_non_replayable_messages,
)


def _message(
    *,
    role: ChatMessageRole,
    value: str,
    sequence: int,
    conversation_id: str = "conversation",
) -> Message:
    message = Message.from_prompt(prompt=value, role=role)
    piece = message.get_piece()
    piece.sequence = sequence
    piece.conversation_id = conversation_id
    return message


def test_context_requires_unique_persisted_history_ids():
    message_id = uuid.uuid4()

    with pytest.raises(ValueError, match="conversation_id"):
        TargetNormalizationContext(
            conversation_id="",
            history_message_ids=(message_id,),
            replay_history_each_send=False,
        )
    with pytest.raises(ValueError, match="history_message_ids"):
        TargetNormalizationContext(
            conversation_id="conversation",
            history_message_ids=(),
            replay_history_each_send=False,
        )
    with pytest.raises(ValueError, match="unique"):
        TargetNormalizationContext(
            conversation_id="conversation",
            history_message_ids=(message_id, message_id),
            replay_history_each_send=False,
        )


def test_stateful_context_consumes_explicit_boundary_at_provider_attempt():
    first = _message(role="system", value="system", sequence=0)
    second = _message(role="user", value="seed", sequence=1)
    unrelated = _message(role="assistant", value="later response", sequence=2)
    context = TargetNormalizationContext(
        conversation_id="conversation",
        history_message_ids=(first.get_piece().id, second.get_piece().id),
        replay_history_each_send=False,
    )

    context.begin_send()
    selected = context.select_history(messages=[unrelated, second, first])
    context.mark_provider_attempted()
    context.finish_send()

    assert selected == [first, second]
    assert context.is_consumed
    assert context.provider_attempt_count == 1

    context.begin_send()
    assert context.select_history(messages=[first, second, unrelated]) == []
    context.finish_send()


def test_stateless_context_replays_explicit_boundary_after_provider_attempt():
    seed = _message(role="user", value="seed", sequence=0)
    context = TargetNormalizationContext(
        conversation_id="conversation",
        history_message_ids=(seed.get_piece().id,),
        replay_history_each_send=True,
    )

    for _ in range(2):
        context.begin_send()
        assert context.select_history(messages=[seed]) == [seed]
        context.mark_provider_attempted()
        context.finish_send()

    assert not context.is_consumed
    assert context.provider_attempt_count == 2


def test_context_rejects_concurrent_send_until_active_send_finishes():
    context = TargetNormalizationContext(
        conversation_id="conversation",
        history_message_ids=(uuid.uuid4(),),
        replay_history_each_send=False,
    )

    context.begin_send()
    with pytest.raises(RuntimeError, match="Concurrent sends"):
        context.begin_send()

    context.finish_send()
    context.begin_send()
    context.finish_send()


def test_context_rejects_provider_attempt_without_active_send():
    context = TargetNormalizationContext(
        conversation_id="conversation",
        history_message_ids=(uuid.uuid4(),),
        replay_history_each_send=False,
    )

    with pytest.raises(RuntimeError, match="without an active send"):
        context.mark_provider_attempted()


def test_context_rejects_missing_persisted_boundary_message():
    context = TargetNormalizationContext(
        conversation_id="conversation",
        history_message_ids=(uuid.uuid4(),),
        replay_history_each_send=False,
    )

    with pytest.raises(ValueError, match="Missing 1 message"):
        context.select_history(messages=[])


@pytest.mark.parametrize("response_error", ["processing", "unknown"])
def test_filter_removes_adjacent_failed_exchange(response_error: PromptResponseError):
    successful = _message(role="user", value="successful", sequence=0)
    failed_request = _message(role="user", value="failed request", sequence=1)
    error_response = _message(role="assistant", value="private stack trace", sequence=2)
    error_response.get_piece().response_error = response_error

    filtered = filter_non_replayable_messages(messages=[successful, failed_request, error_response])

    assert filtered == [successful]


@pytest.mark.parametrize(
    ("preceding_role", "preceding_conversation_id", "preceding_sequence"),
    [
        ("assistant", "conversation", 4),
        ("user", "other-conversation", 4),
        ("user", "conversation", 3),
    ],
)
def test_filter_does_not_remove_unrelated_preceding_message(
    preceding_role: ChatMessageRole,
    preceding_conversation_id: str,
    preceding_sequence: int,
):
    preceding = _message(
        role=preceding_role,
        value="unrelated",
        sequence=preceding_sequence,
        conversation_id=preceding_conversation_id,
    )
    error_response = _message(role="assistant", value="private stack trace", sequence=5)
    error_response.get_piece().response_error = "processing"

    filtered = filter_non_replayable_messages(messages=[preceding, error_response])

    assert filtered == [preceding]


@pytest.mark.parametrize("response_error", ["blocked", "empty"])
def test_filter_retains_provider_round_trip(response_error: PromptResponseError):
    request = _message(role="user", value="request", sequence=0)
    response = _message(role="assistant", value="provider response", sequence=1)
    response.get_piece().response_error = response_error

    filtered = filter_non_replayable_messages(messages=[request, response])

    assert filtered == [request, response]


def test_context_remaps_only_explicit_history_for_duplicate_conversation():
    seed = _message(role="system", value="seed", sequence=0)
    live_request = _message(role="user", value="live", sequence=1)
    source_messages = [seed, live_request]
    duplicated_messages = [message.duplicate() for message in source_messages]
    for message in duplicated_messages:
        for piece in message.message_pieces:
            piece.conversation_id = "duplicate"

    context = TargetNormalizationContext(
        conversation_id="conversation",
        history_message_ids=(seed.get_piece().id,),
        replay_history_each_send=True,
    )

    duplicate = context.remap_for_duplicate_conversation(
        conversation_id="duplicate",
        source_messages=source_messages,
        duplicated_messages=duplicated_messages,
    )

    assert duplicate.history_message_ids == (duplicated_messages[0].get_piece().id,)
    assert duplicated_messages[1].get_piece().id not in duplicate.history_message_ids
    assert duplicate.select_history(messages=duplicated_messages) == [duplicated_messages[0]]


def test_context_remap_preserves_consumed_state():
    seed = _message(role="user", value="seed", sequence=0)
    duplicated_seed = seed.duplicate()
    duplicated_seed.get_piece().conversation_id = "duplicate"
    context = TargetNormalizationContext(
        conversation_id="conversation",
        history_message_ids=(seed.get_piece().id,),
        replay_history_each_send=False,
    )
    context.begin_send()
    context.mark_provider_attempted()
    context.finish_send()

    duplicate = context.remap_for_duplicate_conversation(
        conversation_id="duplicate",
        source_messages=[seed],
        duplicated_messages=[duplicated_seed],
    )

    assert duplicate.is_consumed
    assert duplicate.select_history(messages=[duplicated_seed]) == []
