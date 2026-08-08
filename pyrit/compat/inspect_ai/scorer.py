# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Native result scorer matching Inspect 0.3.233 multiple-choice parsing."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import JSONValue, Score

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyrit.executor.capability import CapabilityTaskResult


def parse_inspect_choice_answer(completion: str, *, allowed_options: frozenset[str]) -> str | None:
    """
    Parse one answer using the pinned Inspect 0.3.233 contract.

    Returns:
        str | None: The normalized answer, or ``None`` when the contract is not met.
    """
    matches = re.findall(
        r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)",
        completion,
        flags=re.MULTILINE,
    )
    if not matches:
        matches = re.findall(
            r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)",
            completion,
        )
    if not matches:
        return None
    matched = matches[-1].strip().rstrip(".").upper()
    return matched if matched in allowed_options else None


class InspectChoiceScorer:
    """Parse and score the pinned Inspect ``ANSWER: <letter>`` response contract."""

    def __init__(
        self,
        *,
        expected_value: str,
        allowed_options: tuple[str, ...],
        memory: MemoryInterface | None = None,
    ) -> None:
        """Initialize the scorer."""
        self._expected_value = expected_value
        self._allowed_options = frozenset(allowed_options)
        self._memory = memory or CentralMemory.get_memory_instance()

    @classmethod
    def from_config(cls, config: Mapping[str, JSONValue]) -> InspectChoiceScorer:
        """
        Build the scorer from a strict manifest configuration.

        Returns:
            InspectChoiceScorer: The configured scorer.

        Raises:
            ValueError: If the scorer configuration is malformed.
        """
        expected = config.get("expected_value")
        options = config.get("allowed_options")
        if not isinstance(expected, str):
            raise ValueError("Inspect choice scorer requires string 'expected_value'.")
        if not isinstance(options, list) or not all(isinstance(item, str) for item in options):
            raise ValueError("Inspect choice scorer requires string-list 'allowed_options'.")
        return cls(
            expected_value=expected,
            allowed_options=tuple(item for item in options if isinstance(item, str)),
        )

    async def score_result_async(self, *, result: CapabilityTaskResult, objective: str) -> list[Score]:
        """
        Score the final completion using the pinned Inspect answer parser.

        Returns:
            list[Score]: One true/false score.
        """
        pieces = self._memory.get_message_pieces(prompt_ids=list(result.final_message_piece_ids))
        by_id = {piece.id: piece for piece in pieces}
        text = "\n".join(
            by_id[piece_id].converted_value
            for piece_id in result.final_message_piece_ids
            if piece_id in by_id and by_id[piece_id].converted_value
        )
        answer = parse_inspect_choice_answer(text, allowed_options=self._allowed_options)
        matched = answer == self._expected_value
        piece_id = result.final_message_piece_ids[-1] if result.final_message_piece_ids else str(result.case_id)
        return [
            Score(
                score_value=str(matched),
                score_type="true_false",
                score_category=[],
                score_rationale="InspectChoiceScorer using inspect_ai 0.3.233 answer parsing.",
                message_piece_id=piece_id,
                objective=objective,
            )
        ]
