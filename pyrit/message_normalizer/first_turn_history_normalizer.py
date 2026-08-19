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
        prepended_message_count: int,
    ) -> None:
        """
        Initialize the normalizer.

        Args:
            message_normalizer: Formatter for the target-facing text.
            prepended_message_count: Number of leading messages that belong to
                the prepended conversation.

        Raises:
            ValueError: If prepended_message_count is less than one.
        """
        if prepended_message_count < 1:
            raise ValueError("prepended_message_count must be at least 1")
        self._message_normalizer = message_normalizer
        self._prepended_message_count = prepended_message_count

    async def normalize_async(self, messages: list[Message]) -> list[Message]:
        """
        Return one target-facing request containing the prepended history.

        Args:
            messages: Prepended history followed by exactly one live request.

        Returns:
            A single request message with formatted text and preserved live
            non-text pieces.

        Raises:
            ValueError: If the input does not contain the configured prepended
                message count followed by one live request, or if converted
                prepended history cannot be represented as text.
        """
        expected_count = self._prepended_message_count + 1
        if len(messages) != expected_count:
            raise ValueError(
                "First-turn history normalization expected "
                f"{self._prepended_message_count} prepended messages and one live request, "
                f"but received {len(messages)} messages."
            )

        prepended_messages = messages[: self._prepended_message_count]
        live_request = messages[-1]
        self._validate_flattenable_converter_output(messages=prepended_messages)

        original_view = self._build_original_view(messages=messages)
        converted_view = self._build_converted_view(messages=messages)
        original_text = await self._normalize_context_async(messages=original_view)
        converted_text = original_text
        if self._contains_converted_values(messages=messages):
            converted_text = await self._normalize_context_async(messages=converted_view)

        return [
            self._build_target_request(
                live_request=live_request,
                original_text=original_text,
                converted_text=converted_text,
            )
        ]

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
