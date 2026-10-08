# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Explicit runtime-owned feedback barriers; attacks still choose and score the next turn."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyrit.memory import MemoryInterface
    from pyrit.models import Message, Score, ScoringExpectation
    from pyrit.prompt_target import PromptTarget
    from pyrit.score import Scorer


class AttackFeedbackObserver(Protocol):
    """An opt-in working-memory observer, not a source grader or prompt-generation algorithm."""

    def validate_setup(
        self,
        *,
        memory: MemoryInterface,
        normalizer_memory: MemoryInterface,
        objective_target: PromptTarget,
        objective_scorer: Scorer,
    ) -> None:
        """Require the target, attack, scorer and observer to share one runtime memory owner."""
        ...

    @property
    def policy_sha256(self) -> str:
        """The declared feedback policy, excluding fresh owner and run identities."""
        ...

    async def before_turn_async(
        self, *, conversation_id: str, turn_index: int, expectation: ScoringExpectation | None
    ) -> None:
        """Reject next generation until the previous observation and required feedback are committed."""
        ...

    async def response_committed_async(self, *, response: Message) -> None:
        """Verify the normalizer's actual committed source-linked messages before scoring."""
        ...

    async def feedback_committed_async(
        self,
        *,
        response: Message,
        scores: Sequence[Score],
        expectation: ScoringExpectation | None,
    ) -> None:
        """Verify the required persisted score/rationale before exposing a ready snapshot."""
        ...

    async def execution_failed_async(self) -> None:
        """Revoke continuation when a writer, scorer or attack fails outside the readback hooks."""
        ...
