# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import uuid

from pyrit.message_normalizer.conversation_context_normalizer import ConversationContextNormalizer
from pyrit.message_normalizer.generic_system_squash import GenericSystemSquashNormalizer
from pyrit.message_normalizer.message_normalizer import MessageListNormalizer, MessageStringNormalizer
from pyrit.models import Message, MessagePiece


class PrependedConversationNormalizer(MessageListNormalizer[Message]):
    """
    Combine prepended history with the first live request for targets without editable history.

    The history remains structured in memory. This normalizer creates an ephemeral target view
    that preserves the live request's modalities while prefixing independently rendered original
    and converted history.
    """

    def __init__(self, *, message_normalizer: MessageStringNormalizer) -> None:
        """
        Initialize the adapter.

        Args:
            message_normalizer: Formatter used to render prepended history.
        """
        self._message_normalizer = message_normalizer

    async def normalize_async(self, messages: list[Message]) -> list[Message]:
        """
        Normalize prepended history into the final request message.

        Args:
            messages: Prepended history followed by the first live request.

        Returns:
            A single request message containing the rendered history.
        """
        if len(messages) < 2:
            return copy.deepcopy(messages)

        prepended_messages = messages[:-1]
        self._validate_flattenable_converter_output(messages=prepended_messages)

        original_context = await self._normalize_context_async(
            messages=self._build_original_view(messages=prepended_messages)
        )
        converted_context = original_context
        if self._contains_converted_values(messages=prepended_messages):
            converted_context = await self._normalize_context_async(
                messages=self._build_converted_view(messages=prepended_messages)
            )

        request = copy.deepcopy(messages[-1])
        self._prepend_context(
            message=request,
            original_context=original_context,
            converted_context=converted_context,
        )
        return [request]

    async def _normalize_context_async(self, *, messages: list[Message]) -> str:
        messages_to_normalize = messages
        if isinstance(self._message_normalizer, ConversationContextNormalizer):
            messages_to_normalize = await GenericSystemSquashNormalizer().normalize_async(messages)
        return await self._message_normalizer.normalize_string_async(messages_to_normalize)

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
            and piece.converted_value_data_type != piece.original_value_data_type
        }
        if output_types:
            raise ValueError(
                "Cannot flatten prepended conversation after request converters produced "
                f"non-text output types {sorted(output_types)}. Prepended conversion must produce "
                "text for a target without editable history."
            )

    @staticmethod
    def _prepend_context(*, message: Message, original_context: str, converted_context: str) -> None:
        text_piece = next(
            (
                piece
                for piece in message.message_pieces
                if piece.original_value_data_type == "text" and piece.converted_value_data_type == "text"
            ),
            None,
        )
        if text_piece:
            text_piece.original_value = PrependedConversationNormalizer._prepend_context_value(
                context=original_context,
                value=text_piece.original_value,
            )
            text_piece.converted_value = PrependedConversationNormalizer._prepend_context_value(
                context=converted_context,
                value=text_piece.converted_value,
            )
            return

        template_piece = message.get_piece()
        message.message_pieces.insert(
            0,
            MessagePiece(
                id=uuid.uuid4(),
                role=template_piece.role,
                original_value=original_context,
                converted_value=converted_context,
                original_value_data_type="text",
                converted_value_data_type="text",
                conversation_id=template_piece.conversation_id,
                sequence=template_piece.sequence,
            ),
        )

    @staticmethod
    def _prepend_context_value(*, context: str, value: str) -> str:
        if not context or value == context or value.startswith(f"{context}\n\n"):
            return value
        return f"{context}\n\n{value}"
