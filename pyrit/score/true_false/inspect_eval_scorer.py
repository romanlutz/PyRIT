# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

from pyrit.models import ComponentIdentifier, MessageScorable, Score
from pyrit.score.message_scorable_resolver import MessageScorableResolver
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from inspect_ai.log import EvalLog, EvalSample

    from pyrit.models import MessagePiece, Scorable, ScoringExpectation


class InspectEvalScorer(TrueFalseScorer):
    """Project a checked native C/I grade onto the real, retained final message."""

    def __init__(
        self,
        *,
        log_path: Path,
        eval_id: str,
        sample_id: int | str,
        sample_uuid: str,
        run_id: str,
        attempt_id: str,
        message_piece_id: UUID,
        input_sha256: str,
        native_scorer: str,
    ) -> None:
        """Bind immutable native and PyRIT evidence identities, not an attack runner."""
        super().__init__()
        self._log_path = log_path
        self._eval_id = eval_id
        self._sample_id = sample_id
        self._sample_uuid = sample_uuid
        self._run_id = run_id
        self._attempt_id = attempt_id
        self._message_piece_id = message_piece_id
        self._input_sha256 = input_sha256
        self._native_scorer = native_scorer
        self._resolver = MessageScorableResolver()

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(
            params={
                "native_scorer": self._native_scorer,
                "eval_id": self._eval_id,
                "sample_id": self._sample_id,
                "sample_uuid": self._sample_uuid,
                "input_sha256": self._input_sha256,
            }
        )

    async def _score_scorable_async(
        self,
        *,
        scorable: Scorable,
        expectation: ScoringExpectation | None,
    ) -> list[Score]:
        from inspect_ai.log import read_eval_log_async

        if not isinstance(scorable, MessageScorable) or scorable.message_piece_ids != (self._message_piece_id,):
            raise TypeError("Inspect grade projection requires its single retained final-message piece.")
        message = await asyncio.to_thread(self._resolver.resolve, scorable=scorable, memory=self._memory)
        piece = message.get_piece()
        self._validate_piece(piece)
        log = await read_eval_log_async(self._log_path, resolve_attachments=True)
        sample = await asyncio.to_thread(self._checked_sample, log)
        if sample.output.completion != piece.converted_value:
            raise ValueError("Native completion does not match the retained PyRIT answer.")
        if sample.metadata.get("pyrit_message_piece_id") != str(piece.id):
            raise ValueError("Native sample does not identify the retained PyRIT message.")
        if sample.metadata.get("pyrit_conversation_id") != piece.conversation_id:
            raise ValueError("Native sample does not identify the retained PyRIT conversation.")
        if not sample.scores or self._native_scorer not in sample.scores:
            raise ValueError("The requested native grade is absent; no PyRIT verdict can be projected.")
        grade = sample.scores[self._native_scorer]
        if grade.value not in ("C", "I"):
            raise ValueError(f"Unsupported native grade {grade.value!r}; this prototype only maps C/I.")
        return [
            Score(
                score_type="true_false",
                score_value=str(grade.value == "C").lower(),
                score_rationale=f"Retained Inspect scorer {self._native_scorer} returned {grade.value}.",
                score_category=["task_correctness"],
                scorer_class_identifier=self.get_identifier(),
                message_piece_id=piece.id,
                scorable=scorable,
                scored_expectation=expectation,
                score_metadata={
                    "native_grade": str(grade.value),
                    "native_eval_id": self._eval_id,
                    "native_sample_uuid": self._sample_uuid,
                    "native_log": str(self._log_path),
                    "run_id": self._run_id,
                    "attempt_id": self._attempt_id,
                    "input_sha256": self._input_sha256,
                },
            )
        ]

    def _checked_sample(self, log: EvalLog) -> EvalSample:
        from inspect_ai.log import EvalSample

        if log.status != "success" or log.error or log.invalidated or log.eval.eval_id != self._eval_id:
            raise ValueError("Native evaluation is incomplete, errored, invalidated, or has the wrong identity.")
        metadata = log.eval.metadata or {}
        if metadata.get("pyrit_run_id") != self._run_id or metadata.get("pyrit_attempt_id") != self._attempt_id:
            raise ValueError("Native evaluation is not bound to this run and independent attempt.")
        samples = list(log.samples or [])
        if len(samples) != 1:
            raise ValueError("Expected exactly one native sample and epoch.")
        sample = EvalSample.model_validate(samples[0])
        if sample.id != self._sample_id or sample.epoch != 1 or sample.uuid != self._sample_uuid:
            raise ValueError("Native sample/epoch identity does not match the bound evidence.")
        if sample.error or sample.error_retries or sample.invalidation or sample.output.error:
            raise ValueError("An errored, retried, or invalidated sample cannot supply a correctness verdict.")
        if (
            not isinstance(sample.input, str)
            or hashlib.sha256(sample.input.encode("utf-8")).hexdigest() != self._input_sha256
        ):
            raise ValueError("Native task input has changed.")
        return sample

    @staticmethod
    def _validate_piece(piece: MessagePiece) -> None:
        if (
            piece.api_role != "assistant"
            or piece.is_simulated
            or piece.converted_value_data_type != "text"
            or piece.has_error()
            or piece.is_truncated
            or not piece.converted_value.strip()
        ):
            raise ValueError("Native grading requires an authentic, complete, nonempty assistant text answer.")
