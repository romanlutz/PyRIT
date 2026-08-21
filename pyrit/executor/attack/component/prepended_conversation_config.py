# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyrit.message_normalizer import (
    ConversationContextNormalizer,
    FirstTurnHistoryNormalizer,
    MessageListNormalizer,
    MessageStringNormalizer,
)
from pyrit.prompt_target.common.target_capabilities import CapabilityName

if TYPE_CHECKING:
    from pyrit.models import ChatMessageRole, Message
    from pyrit.prompt_target.common.prompt_target import PromptTarget


@dataclass
class PrependedConversationConfig:
    """
    Configuration for controlling how prepended conversations are processed before
    being sent to the objective target.

    This class provides control over:
    - Which message roles should have request converters applied
    - How targets without editable history format prepended messages with live requests

    Prepended messages remain role-structured in memory. Request converters are applied to
    configured roles before a target without editable history renders that history and the
    applicable live request together (via ``message_normalizer``; default: ConversationContextNormalizer).
    Those converters must produce text because string normalization cannot preserve converted
    image, audio, or other non-text output.
    """

    # Request converters default to prepended user messages only. Assistant history is
    # simulated target output and must be explicitly opted in with ["assistant"].
    apply_converters_to_roles: list[ChatMessageRole] = field(default_factory=lambda: ["user"])

    # Optional normalizer to format prepended history and a live request as one text block.
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

    def get_normalizer_overrides(
        self,
        *,
        target: PromptTarget,
    ) -> dict[CapabilityName, MessageListNormalizer[Message]]:
        """
        Build per-send target normalizer overrides for prepended history.

        Args:
            target: Target that receives the live request.

        Returns:
            Overrides keyed by the capability they adapt.
        """
        if target.configuration.includes(capability=CapabilityName.EDITABLE_HISTORY):
            return {}

        return {
            CapabilityName.EDITABLE_HISTORY: FirstTurnHistoryNormalizer(
                message_normalizer=self.get_message_normalizer(),
                target_supports_multi_turn=target.configuration.includes(capability=CapabilityName.MULTI_TURN),
            )
        }
