# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pyrit.memory import CentralMemory
from pyrit.models import Message, MessageScorable
from pyrit.prompt_normalizer import PromptNormalizer

if TYPE_CHECKING:
    from pyrit.models import Score, SeedPrompt
    from pyrit.prompt_target import PromptTarget
    from pyrit.score import Scorer


class NativeCTFBenchmark:
    """Send a prepared seed through PyRIT and score only a real final assistant message."""

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        scorer: Scorer,
        prompt_normalizer: PromptNormalizer | None = None,
    ) -> None:
        """Compose public target, normalizer, memory, and scorer APIs."""
        self._target = objective_target
        self._scorer = scorer
        self._normalizer = prompt_normalizer or PromptNormalizer()
        self._memory = CentralMemory.get_memory_instance()

    async def send_async(self, *, seed: SeedPrompt, system_prompt: str, conversation_id: str, run_id: str) -> Message:
        """
        Send one prepared task, preserving the target's actual tool loop.

        Returns:
            Message: The authentic final message, not a request echoed by a write-only target.

        Raises:
            ValueError: If the target did not produce one complete final assistant text.
        """
        await self._memory.add_seeds_to_memory_async(seeds=[seed], added_by="native_ctf")
        await asyncio.to_thread(
            self._target.set_system_prompt, system_prompt=system_prompt, conversation_id=conversation_id
        )
        response = await self._normalizer.send_prompt_async(
            message=Message.from_prompt(
                prompt=seed.value,
                role="user",
                prompt_metadata={"native_ctf_run_id": run_id, "seed_id": str(seed.id)},
            ),
            target=self._target,
            conversation_id=conversation_id,
        )
        if (
            response.api_role != "assistant"
            or response.is_simulated
            or any(piece.has_error() for piece in response.message_pieces)
        ):
            raise ValueError("The target did not return an authentic, error-free assistant answer.")
        texts = response.get_pieces_by_type(data_type="text")
        if len(texts) != 1 or not texts[0].converted_value.strip():
            raise ValueError("The target did not return exactly one nonempty final text.")
        if any(
            piece.is_truncated or piece.converted_value_data_type not in {"text", "reasoning"}
            for piece in response.message_pieces
        ):
            raise ValueError("An incomplete response or a pending tool call is not a final answer.")
        return response

    async def score_async(self, response: Message) -> Score:
        """
        Persist a score anchored to the real final message.

        Returns:
            Score: The scorer's single completed verdict.

        Raises:
            ValueError: If the scorer cannot produce a single completed verdict.
        """
        scores = await self._scorer.score_async(scorable=MessageScorable.from_message(response))
        if len(scores) != 1 or scores[0].score_value is None:
            raise ValueError("The final answer did not yield a completed score.")
        return scores[0]
