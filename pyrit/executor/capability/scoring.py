# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Evidence-aware scoring seams for capability-task results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import Message

if TYPE_CHECKING:
    from pyrit.executor.capability.models import CapabilityTaskResult
    from pyrit.models import Score
    from pyrit.score import Scorer


class CapabilityResultScorer(Protocol):
    """A scorer that can consume a complete capability result and evidence."""

    async def score_result_async(self, *, result: CapabilityTaskResult, objective: str) -> list[Score]:
        """Score a capability result without owning executor branching."""


class MessageScorerAdapter:
    """Adapt an existing message scorer to the capability-result scoring seam."""

    def __init__(self, *, scorer: Scorer, memory: MemoryInterface | None = None) -> None:
        """Initialize the adapter."""
        self._scorer = scorer
        self._memory = memory or CentralMemory.get_memory_instance()

    async def score_result_async(self, *, result: CapabilityTaskResult, objective: str) -> list[Score]:
        """
        Score the final assistant message referenced by the result.

        Returns:
            list[Score]: Scores produced and persisted by the existing scorer.
        """
        pieces = self._memory.get_message_pieces(prompt_ids=list(result.final_message_piece_ids))
        if not pieces:
            return []
        by_id = {piece.id: piece for piece in pieces}
        ordered = [by_id[piece_id] for piece_id in result.final_message_piece_ids if piece_id in by_id]
        return await self._scorer.score_async(Message(message_pieces=ordered), objective=objective)
