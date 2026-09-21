# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Local targets for seeded CoPyRIT tests; no external model or credentials."""

import asyncio

from pyrit.models import Message, TargetCapabilities, construct_response_from_request
from pyrit.prompt_target import PromptTarget, TargetConfiguration
from pyrit.registry import TargetRegistry
from pyrit.setup.pyrit_initializer import PyRITInitializer


class FrontendEchoTarget(PromptTarget):
    """Return explicitly labeled offline echoes through the normal target and memory paths."""

    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_multi_message_pieces=True,
            supports_system_prompt=True,
            supports_editable_history=True,
        )
    )

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        """Generate a local reply without bypassing the normalizer."""
        request = normalized_conversation[-1]
        text = request.get_value()
        await asyncio.sleep(0.1)
        return [
            construct_response_from_request(
                request=request.message_pieces[0],
                response_text_pieces=[f"Offline test response: {text}"],
                response_type="text",
            )
        ]

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        """Require the request that the fixture will answer."""
        if not normalized_conversation:
            raise ValueError("A local echo request must contain a message")


class FrontendTargetsInitializer(PyRITInitializer):
    """Register deterministic local targets for offline browser tests."""

    async def initialize_async(self) -> None:
        """Register the shared seeded-test target."""
        TargetRegistry.get_registry_singleton().instances.register(
            instance=FrontendEchoTarget(),
            name="frontend_echo",
        )
