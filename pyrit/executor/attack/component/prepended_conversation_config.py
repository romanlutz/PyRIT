# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyrit.message_normalizer import (
    ConversationContextNormalizer,
    MessageStringNormalizer,
)

if TYPE_CHECKING:
    from pyrit.models import ChatMessageRole


@dataclass
class PrependedConversationConfig:
    """
    Configuration for controlling how prepended conversations are processed before
    being sent to the objective target.

    This class provides control over:
    - Which message roles should have request converters applied
    - How targets without editable history format prepended messages on the first live send

    Prepended messages remain role-structured in memory. Request converters are applied to
    configured roles before a target without editable history renders that history into the
    first live request (via ``message_normalizer``; default: ConversationContextNormalizer).
    Those converters must produce text because string normalization cannot preserve converted
    image, audio, or other non-text output.
    """

    # Request converters default to prepended user messages only. Assistant history is
    # simulated target output and must be explicitly opted in with ["assistant"].
    apply_converters_to_roles: list[ChatMessageRole] = field(default_factory=lambda: ["user"])

    # Optional normalizer to format conversation history into a single text block.
    # Must implement MessageStringNormalizer (e.g., TokenizerTemplateNormalizer or ConversationContextNormalizer).
    # When None and adaptation is needed, a default ConversationContextNormalizer is used
    # that produces "Turn N: User/Assistant" format.
    message_normalizer: MessageStringNormalizer | None = None

    def get_message_normalizer(self) -> MessageStringNormalizer:
        """
        Get the normalizer for objective target context, with a default fallback.

        Returns:
            The configured objective_target_context_normalizer, or a default
            ConversationContextNormalizer if none was configured.
        """
        return self.message_normalizer or ConversationContextNormalizer()
