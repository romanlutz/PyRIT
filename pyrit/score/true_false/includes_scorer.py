# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import hashlib

from pyrit.models import ComponentIdentifier, MessagePiece, Score
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_scorer import MessageTrueFalseScorer


class IncludesScorer(MessageTrueFalseScorer):
    """Match Inspect's includes() semantics without importing Inspect."""

    def __init__(self, *, expected: str, ignore_case: bool = True, categories: list[str] | None = None) -> None:
        """
        Initialize trusted grading text; it is never sent to a model.

        Args:
            expected (str): Trusted text to look for.
            ignore_case (bool): Whether to casefold both strings before matching.
            categories (list[str] | None): Caller-owned score categories.
        """
        self._expected = expected
        self._ignore_case = ignore_case
        self._score_categories = list(categories or [])
        super().__init__(validator=ScorerPromptValidator(supported_data_types=["text"], supported_roles=["assistant"]))

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(
            params={
                "expected_sha256": hashlib.sha256(self._expected.encode("utf-8")).hexdigest(),
                "ignore_case": self._ignore_case,
                "matching": "casefold-substring" if self._ignore_case else "substring",
                "categories": self._score_categories,
            }
        )

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: str | None = None) -> list[Score]:
        answer = message_piece.converted_value
        expected = self._expected
        if self._ignore_case:
            answer, expected = answer.casefold(), expected.casefold()
        matched = expected in answer
        return [
            Score(
                score_value=str(matched).lower(),
                score_type="true_false",
                score_category=self._score_categories,
                message_piece_id=message_piece.id,
                scorer_class_identifier=self.get_identifier(),
                objective=objective,
                score_metadata={
                    "value": "C" if matched else "I",
                    "answer": answer,
                    "explanation": message_piece.converted_value,
                },
                score_rationale="Casefold substring inclusion, without trimming whitespace or punctuation."
                if self._ignore_case
                else "Case-sensitive substring inclusion.",
            )
        ]
