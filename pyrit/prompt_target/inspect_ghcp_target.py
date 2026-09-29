# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A retained GHCP session owned by one Inspect sample's agent sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from pyrit.models import Message, construct_response_from_request
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_capabilities import (
    CapabilityHandlingPolicy,
    CapabilityName,
    TargetCapabilities,
    UnsupportedCapabilityBehavior,
)
from pyrit.prompt_target.common.target_configuration import TargetConfiguration

if TYPE_CHECKING:
    from uuid import UUID

    from pydantic import JsonValue

    from pyrit.models import ComponentIdentifier


@dataclass(frozen=True, kw_only=True)
class InspectGhcpTurn:
    """The actual SDK turn and the exact MessagePieces persisted by PyRIT."""

    turn_index: int
    session_id: str
    identity: dict[str, JsonValue]
    instruction: str
    assistant_text: str
    request_piece_id: UUID
    response_piece_id: UUID
    events: tuple[dict[str, JsonValue], ...]
    model_exchanges: tuple[dict[str, JsonValue], ...]


class InspectGhcpTransport(Protocol):
    """Send text to the already-running, sandbox-contained SDK session."""

    async def send_turn_async(self, *, instruction: str, turn_index: int) -> dict[str, JsonValue]:
        """Return the source-observed SDK frame without synthesizing an answer."""
        ...


class InspectGhcpTarget(PromptTarget):
    """Send prepared text turns into one retained Inspect-owned GHCP sandbox session."""

    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_multi_message_pieces=False,
            supports_system_prompt=False,
            supports_editable_history=False,
        ),
        policy=CapabilityHandlingPolicy(
            behaviors={
                CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.RAISE,
                CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.RAISE,
                CapabilityName.JSON_SCHEMA: UnsupportedCapabilityBehavior.RAISE,
            }
        ),
    )

    def __init__(self, *, transport: InspectGhcpTransport, run_id: str, model_name: str) -> None:
        """Bind a caller-owned sandbox transport without opening a new session."""
        super().__init__(endpoint="inspect://agent", model_name=model_name)
        self._transport = transport
        self._run_id = run_id
        self._conversation_id: str | None = None
        self._turns: list[InspectGhcpTurn] = []

    @property
    def turns(self) -> tuple[InspectGhcpTurn, ...]:
        """The observed sends and genuine PyRIT MessagePiece identifiers."""
        return tuple(self._turns)

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(params={"adapter": "inspect_ghcp", "run_id": self._run_id})

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        """
        Send only the last new user instruction to the same SDK process and session.

        Returns:
            list[Message]: The observed assistant message retained by the normalizer.

        Raises:
            ValueError: If the input edits/replays history or the SDK frame is incomplete.
        """
        request = self._validate_history(normalized_conversation=normalized_conversation)
        request_piece = request.get_piece()
        frame = await self._transport.send_turn_async(
            instruction=request_piece.converted_value, turn_index=len(self._turns) + 1
        )
        turn_index = len(self._turns) + 1
        session_id = frame.get("session_id")
        identity = frame.get("identity")
        answer = frame.get("assistant_text")
        events = frame.get("events")
        exchanges = frame.get("model_exchanges")
        if (
            frame.get("kind") != "turn"
            or frame.get("run_id") != self._run_id
            or frame.get("turn_index") != turn_index
            or not isinstance(session_id, str)
            or not session_id
            or not isinstance(identity, dict)
            or not isinstance(answer, str)
            or not answer
            or not isinstance(events, list)
            or not isinstance(exchanges, list)
            or not all(isinstance(event, dict) for event in events)
            or not all(isinstance(exchange, dict) for exchange in exchanges)
        ):
            raise ValueError("The contained GHCP worker returned an incomplete or foreign turn.")
        if self._turns and (session_id != self._turns[0].session_id or identity != self._turns[0].identity):
            raise ValueError("The GHCP session or observed agent process changed between PyRIT turns.")
        response = construct_response_from_request(
            request=request_piece,
            response_text_pieces=[answer],
            prompt_metadata={"inspect_ghcp_session_id": session_id, "inspect_ghcp_turn": turn_index},
        )
        self._turns.append(
            InspectGhcpTurn(
                turn_index=turn_index,
                session_id=session_id,
                identity=identity,
                instruction=request_piece.converted_value,
                assistant_text=answer,
                request_piece_id=request_piece.id,
                response_piece_id=response.get_piece().id,
                events=tuple(events),
                model_exchanges=tuple(exchanges),
            )
        )
        return [response]

    def _validate_history(self, *, normalized_conversation: list[Message]) -> Message:
        if self.configuration is not type(self)._DEFAULT_CONFIGURATION:
            raise ValueError("Inspect GHCP target capabilities cannot be overridden.")
        if len(normalized_conversation) != 2 * len(self._turns) + 1:
            raise ValueError("Inspect GHCP cannot replay, prepend, or edit its retained SDK history.")
        for index, turn in enumerate(self._turns):
            user, assistant = normalized_conversation[2 * index : 2 * index + 2]
            if user.get_value() != turn.instruction or assistant.get_value() != turn.assistant_text:
                raise ValueError("PyRIT history differs from the retained GHCP session.")
        request = normalized_conversation[-1]
        if request.api_role != "user" or len(request.message_pieces) != 1:
            raise ValueError("Inspect GHCP requires exactly one prepared user text piece.")
        piece = request.get_piece()
        if piece.converted_value_data_type != "text" or not piece.converted_value.strip():
            raise ValueError("Inspect GHCP does not accept empty text, media or tool messages.")
        if not piece.conversation_id:
            raise ValueError("PyRIT must assign the GHCP conversation before the target send.")
        if self._conversation_id is not None and piece.conversation_id != self._conversation_id:
            raise ValueError("A second PyRIT conversation cannot inherit the first GHCP session.")
        if piece.prompt_metadata.get("response_format") == "json" or "json_schema" in piece.prompt_metadata:
            raise ValueError("Inspect GHCP does not support target-side structured output.")
        self._conversation_id = piece.conversation_id
        return request
