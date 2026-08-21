# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import uuid

from pyrit.message_normalizer.conversation_context_normalizer import ConversationContextNormalizer
from pyrit.message_normalizer.generic_system_squash import GenericSystemSquashNormalizer
from pyrit.message_normalizer.message_normalizer import MessageListNormalizer, MessageStringNormalizer
from pyrit.models import Message, MessagePiece


class FirstTurnHistoryNormalizer(MessageListNormalizer[Message]):
    """
    Combine structured prepended history with the first live request.

    The configured string normalizer receives both the prepended history and
    the live request, so tokenizer templates place generation markers after the
    live user content. Non-text live pieces remain separate in the target view.
    """

    def __init__(
        self,
        *,
        message_normalizer: MessageStringNormalizer,
        target_supports_multi_turn: bool,
    ) -> None:
        """
        Initialize the normalizer.

        Args:
            message_normalizer: Formatter for the target-facing text.
            target_supports_multi_turn: Whether the target retains state across
                live requests.
        """
        self._message_normalizer = message_normalizer
        self._target_supports_multi_turn = target_supports_multi_turn

    async def normalize_async(self, messages: list[Message]) -> list[Message]:
        """
        Adapt memory-backed history for a target whose history is not editable.

        Before the target has replied, persisted prepended history is formatted
        with the first live request. Failed live request/error pairs are excluded.
        After a real target reply, a stateful target receives only the current
        request. A stateless target receives the original prepended prefix plus
        the current request on every send.

        Args:
            messages: Persisted conversation history followed by the current request.

        Returns:
            A single request message with formatted text and preserved live
            non-text pieces.

        Raises:
            ValueError: If messages is empty or converted prepended history
                cannot be represented as text.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        if len(messages) == 1:
            return list(messages)

        history = self._remove_failed_live_requests(messages=messages[:-1])
        messages = [*history, messages[-1]]
        if len(messages) == 1:
            return messages

        first_response_index = self._find_first_target_response_index(messages=history)
        if first_response_index is None:
            messages_to_format = messages
        elif self._target_supports_multi_turn:
            return [messages[-1]]
        else:
            prepended_end = max(first_response_index - 1, 0)
            prepended_messages = messages[:prepended_end]
            if not prepended_messages:
                return [messages[-1]]
            messages_to_format = [*prepended_messages, messages[-1]]

        prepended_messages = messages_to_format[:-1]
        live_request = messages_to_format[-1]
        self._validate_flattenable_converter_output(messages=prepended_messages)

        original_view = self._build_original_view(messages=messages_to_format)
        converted_view = self._build_converted_view(messages=messages_to_format)
        original_text = await self._normalize_context_async(messages=original_view)
        converted_text = original_text
        if self._contains_converted_values(messages=messages_to_format):
            converted_text = await self._normalize_context_async(messages=converted_view)

        return [
            self._build_target_request(
                live_request=live_request,
                original_text=original_text,
                converted_text=converted_text,
            )
        ]

    @staticmethod
    def _find_first_target_response_index(*, messages: list[Message]) -> int | None:
        for index, message in enumerate(messages):
            if any(piece.role == "assistant" and piece.response_error == "none" for piece in message.message_pieces):
                return index
        return None

    @staticmethod
    def _remove_failed_live_requests(*, messages: list[Message]) -> list[Message]:
        history = list(messages)
        while len(history) >= 2 and any(
            piece.role == "assistant" and piece.response_error != "none" for piece in history[-1].message_pieces
        ):
            history = history[:-2]
        return history

    async def _normalize_context_async(self, *, messages: list[Message]) -> str:
        messages_to_normalize = self._filter_live_non_text_pieces(messages=messages)
        if isinstance(self._message_normalizer, ConversationContextNormalizer):
            messages_to_normalize = await GenericSystemSquashNormalizer().normalize_async(messages_to_normalize)
        return await self._message_normalizer.normalize_string_async(messages_to_normalize)

    @staticmethod
    def _filter_live_non_text_pieces(*, messages: list[Message]) -> list[Message]:
        filtered = copy.deepcopy(messages)
        live_message = filtered[-1]
        text_pieces = [piece for piece in live_message.message_pieces if piece.converted_value_data_type == "text"]
        if not text_pieces:
            filtered.pop()
        else:
            live_message.message_pieces = text_pieces
        return filtered

    @staticmethod
    def _build_original_view(*, messages: list[Message]) -> list[Message]:
        original_messages = copy.deepcopy(messages)
        for message in original_messages:
            for piece in message.message_pieces:
                piece.converted_value = piece.original_value
                piece.converted_value_data_type = piece.original_value_data_type
        return original_messages

    @staticmethod
    def _build_converted_view(*, messages: list[Message]) -> list[Message]:
        converted_messages = copy.deepcopy(messages)
        for message in converted_messages:
            for piece in message.message_pieces:
                piece.original_value = piece.converted_value
                piece.original_value_data_type = piece.converted_value_data_type
        return converted_messages

    @staticmethod
    def _contains_converted_values(*, messages: list[Message]) -> bool:
        return any(
            piece.converter_identifiers
            or piece.original_value != piece.converted_value
            or piece.original_value_data_type != piece.converted_value_data_type
            for message in messages
            for piece in message.message_pieces
        )

    @staticmethod
    def _validate_flattenable_converter_output(*, messages: list[Message]) -> None:
        output_types = {
            piece.converted_value_data_type
            for message in messages
            for piece in message.message_pieces
            if piece.converted_value_data_type != "text"
            and (
                piece.original_value != piece.converted_value
                or piece.original_value_data_type != piece.converted_value_data_type
            )
        }
        if output_types:
            raise ValueError(
                "Cannot flatten prepended conversation after request converters produced "
                f"non-text output types {sorted(output_types)}. Prepended conversion must produce "
                "text for a target without editable history."
            )

    @staticmethod
    def _build_target_request(
        *,
        live_request: Message,
        original_text: str,
        converted_text: str,
    ) -> Message:
        request = copy.deepcopy(live_request)
        template_piece = request.get_piece()
        text_piece = MessagePiece(
            id=uuid.uuid4(),
            role=template_piece.role,
            original_value=original_text,
            converted_value=converted_text,
            original_value_data_type="text",
            converted_value_data_type="text",
            conversation_id=template_piece.conversation_id,
            sequence=template_piece.sequence,
            prompt_metadata=dict(template_piece.prompt_metadata),
        )
        target_pieces: list[MessagePiece] = []
        text_inserted = False
        for piece in request.message_pieces:
            if piece.converted_value_data_type == "text":
                if not text_inserted:
                    target_pieces.append(text_piece)
                    text_inserted = True
                continue
            target_pieces.append(piece)
        if not text_inserted:
            target_pieces.insert(0, text_piece)
        request.message_pieces = target_pieces
        return request
