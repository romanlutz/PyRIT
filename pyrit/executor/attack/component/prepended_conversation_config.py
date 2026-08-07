# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass, field

from pyrit.message_normalizer import (
    ConversationContextNormalizer,
    MessageStringNormalizer,
)
from pyrit.models import ChatMessageRole  # noqa: TC001 - public annotation must resolve at runtime


@dataclass
class PrependedConversationConfig:
    """
    Configuration for controlling how prepended conversations are processed before
    being sent to the objective target.

    This class provides control over:
    - Which message roles should have request converters applied
    - How to normalize conversation history for non-chat objective targets

    Non-chat objective targets always normalize the prepended conversation into the
    first turn (via ``message_normalizer``; default: ConversationContextNormalizer).
    """

    # Request converters default to prepended user messages only. Assistant history is
    # simulated target output and must be explicitly opted in with ["assistant"].
    apply_converters_to_roles: list[ChatMessageRole] = field(default_factory=lambda: ["user"])

    # Optional normalizer to format conversation history into a single text block.
    # Must implement MessageStringNormalizer (e.g., TokenizerTemplateNormalizer or ConversationContextNormalizer).
    # When None and normalization is needed (e.g., for non-chat targets), a default
    # ConversationContextNormalizer is used that produces "Turn N: User/Assistant" format.
    message_normalizer: MessageStringNormalizer | None = None

    def get_message_normalizer(self) -> MessageStringNormalizer:
        """
        Get the normalizer for objective target context, with a default fallback.

        Returns:
            The configured objective_target_context_normalizer, or a default
            ConversationContextNormalizer if none was configured.
        """
        return self.message_normalizer or ConversationContextNormalizer()
