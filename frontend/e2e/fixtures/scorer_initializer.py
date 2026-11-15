# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Resolve catalog scorer defaults locally without enabling provider-backed scoring."""

from pyrit.models import Message
from pyrit.prompt_target import PromptTarget, TargetCapabilities, TargetConfiguration
from pyrit.registry import TargetRegistry
from pyrit.setup.pyrit_initializer import PyRITInitializer


class OfflineScorerTarget(PromptTarget):
    """Support scorer metadata only; fixture execution must supply a local scorer."""

    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(supports_multi_turn=True, supports_system_prompt=True)
    )

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        raise RuntimeError("Offline browser fixtures must supply a local scorer before scoring responses.")

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        if not normalized_conversation:
            raise ValueError("The offline scorer target requires a message.")


class OfflineScorerInitializer(PyRITInitializer):
    """Register a provider-free target before scenario metadata resolves scorer defaults."""

    async def initialize_async(self) -> None:
        TargetRegistry.get_registry_singleton().instances.register(
            OfflineScorerTarget(), name="objective_scorer_chat"
        )
