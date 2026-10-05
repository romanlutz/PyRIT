# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Real PromptNormalizer/SQLite sends into one retained GHCP target session."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.models import Message
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.inspect_ghcp_target import InspectGhcpTarget, InspectGhcpTransport

if TYPE_CHECKING:
    from pydantic import JsonValue

    from pyrit.memory import SQLiteMemory


@pytest.mark.usefixtures("patch_central_database")
class TestInspectGhcpTarget:
    async def test_two_actual_persisted_turns_use_one_source_session(self, sqlite_instance: SQLiteMemory) -> None:
        identity = {"worker_pid": 200, "cli_pid": 201, "uid": 10001, "net_namespace": "net:[1]"}
        transport = MagicMock(spec=InspectGhcpTransport)
        transport.send_turn_async = AsyncMock(
            side_effect=[
                {
                    "kind": "turn",
                    "run_id": "inspect-episode",
                    "turn_index": 1,
                    "session_id": "same-sdk-session",
                    "identity": identity,
                    "assistant_text": "first answer",
                    "events": [{"id": "first", "type": "assistant.message", "data": {"content": "first answer"}}],
                    "model_exchanges": [{"request_id": "first-model"}],
                },
                {
                    "kind": "turn",
                    "run_id": "inspect-episode",
                    "turn_index": 2,
                    "session_id": "same-sdk-session",
                    "identity": identity,
                    "assistant_text": "second answer",
                    "events": [{"id": "second", "type": "assistant.message", "data": {"content": "second answer"}}],
                    "model_exchanges": [{"request_id": "second-model"}],
                },
            ]
        )
        target = InspectGhcpTarget(transport=transport, run_id="inspect-episode", model_name="approved")
        normalizer = PromptNormalizer()
        first = await normalizer.send_prompt_async(
            message=Message.from_prompt(prompt="first instruction", role="user"),
            target=target,
            conversation_id="one-pyrit-conversation",
        )
        second = await normalizer.send_prompt_async(
            message=Message.from_prompt(prompt="PyRIT adversarial follow-up", role="user"),
            target=target,
            conversation_id="one-pyrit-conversation",
        )
        assert (first.get_value(), second.get_value()) == ("first answer", "second answer")
        assert [turn.instruction for turn in target.turns] == [
            "first instruction",
            "PyRIT adversarial follow-up",
        ]
        assert len({turn.session_id for turn in target.turns}) == 1
        assert [call.kwargs["turn_index"] for call in transport.send_turn_async.call_args_list] == [1, 2]
        pieces = sqlite_instance.get_message_pieces(conversation_id="one-pyrit-conversation")
        assert len(pieces) == 4
        assert target.turns[-1].response_piece_id in {piece.id for piece in pieces}

    async def test_rejects_a_foreign_source_session(self, sqlite_instance: SQLiteMemory) -> None:
        transport = MagicMock(spec=InspectGhcpTransport)
        transport.send_turn_async = AsyncMock(
            side_effect=[
                {
                    "kind": "turn",
                    "run_id": "inspect-episode",
                    "turn_index": 1,
                    "session_id": "session-one",
                    "identity": {"cli_pid": 1},
                    "assistant_text": "answer",
                    "events": [],
                    "model_exchanges": [],
                },
                {
                    "kind": "turn",
                    "run_id": "inspect-episode",
                    "turn_index": 2,
                    "session_id": "session-two",
                    "identity": {"cli_pid": 2},
                    "assistant_text": "answer",
                    "events": [],
                    "model_exchanges": [],
                },
            ]
        )
        target = InspectGhcpTarget(transport=transport, run_id="inspect-episode", model_name="approved")
        normalizer = PromptNormalizer()
        await normalizer.send_prompt_async(
            message=Message.from_prompt(prompt="first", role="user"),
            target=target,
            conversation_id="same-conversation",
        )
        with pytest.raises(Exception, match="Error sending prompt"):
            await normalizer.send_prompt_async(
                message=Message.from_prompt(prompt="second", role="user"),
                target=target,
                conversation_id="same-conversation",
            )
        assert len(target.turns) == 1

    @pytest.mark.parametrize("field", ["events", "model_exchanges"])
    async def test_rejects_nonobject_evidence_without_filtering_it_async(self, field: str) -> None:
        frame: dict[str, JsonValue] = {
            "kind": "turn",
            "run_id": "inspect-episode",
            "turn_index": 1,
            "session_id": "same-session",
            "identity": {"cli_pid": 1},
            "assistant_text": "answer",
            "events": [{}],
            "model_exchanges": [{}],
        }
        frame[field] = ["not a structured object"]
        transport = MagicMock(spec=InspectGhcpTransport)
        transport.send_turn_async = AsyncMock(return_value=frame)
        target = InspectGhcpTarget(transport=transport, run_id="inspect-episode", model_name="approved")
        with pytest.raises(Exception, match="Error sending prompt"):
            await PromptNormalizer().send_prompt_async(
                message=Message.from_prompt(prompt="fixture", role="user"),
                target=target,
                conversation_id="source-conversation",
            )
        assert target.turns == ()
