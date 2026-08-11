# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Native result scorer matching Inspect 0.3.233 multiple-choice parsing."""

from __future__ import annotations

import re
import string
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from pyrit.compat.inspect_ai.types import Score as InspectScore
from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import JSONValue, Score
from pyrit.sandbox import SandboxExecRequest, SandboxOperationStatus

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Mapping

    from pyrit.executor.capability import CapabilityTaskResult
    from pyrit.sandbox import SandboxSession


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
        piece_id = str(result.final_message_piece_ids[-1]) if result.final_message_piece_ids else str(result.case_id)
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


class InspectTextScorerConfig(BaseModel):
    """Strict configuration for native Inspect text matching."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_values: tuple[str, ...]
    mode: Literal["includes", "match"]
    location: Literal["begin", "end", "any", "exact"] = "end"
    ignore_case: bool = True


class InspectTextScorer:
    """Native execution of Inspect ``match`` and ``includes`` scorer semantics."""

    def __init__(
        self,
        *,
        config: InspectTextScorerConfig,
        memory: MemoryInterface | None = None,
    ) -> None:
        """Initialize the scorer."""
        self._config = config
        self._memory = memory or CentralMemory.get_memory_instance()

    @classmethod
    def from_config(cls, config: Mapping[str, JSONValue]) -> InspectTextScorer:
        """
        Build the scorer from a strict declarative configuration.

        Returns:
            InspectTextScorer: The configured scorer.

        Raises:
            ValueError: If the scorer mode or match location is unsupported.
        """
        parsed = InspectTextScorerConfig.model_validate(dict(config))
        if parsed.mode not in {"includes", "match"}:
            raise ValueError(f"Unsupported Inspect text scorer mode '{parsed.mode}'.")
        if parsed.location not in {"begin", "end", "any", "exact"}:
            raise ValueError(f"Unsupported Inspect match location '{parsed.location}'.")
        return cls(config=parsed)

    async def score_result_async(self, *, result: CapabilityTaskResult, objective: str) -> list[Score]:
        """
        Score the final completion against one or more accepted targets.

        Returns:
            list[Score]: One normalized native score.
        """
        pieces = self._memory.get_message_pieces(prompt_ids=list(result.final_message_piece_ids))
        by_id = {piece.id: piece for piece in pieces}
        completion = "\n".join(
            by_id[piece_id].converted_value
            for piece_id in result.final_message_piece_ids
            if piece_id in by_id and by_id[piece_id].converted_value
        )
        matched = any(
            self._matches(completion=completion, expected=expected) for expected in self._config.expected_values
        )
        piece_id = str(result.final_message_piece_ids[-1]) if result.final_message_piece_ids else str(result.case_id)
        return normalize_inspect_score(
            score=InspectScore(
                value=matched,
                answer=completion,
                explanation=f"Inspect {self._config.mode} scorer.",
            ),
            message_piece_id=piece_id,
            objective=objective,
        )

    def _matches(self, *, completion: str, expected: str) -> bool:
        left, right = completion, expected
        if self._config.mode == "match":
            left = left.strip()
            right = right.strip()
        if self._config.ignore_case:
            left, right = left.casefold(), right.casefold()
        if self._config.mode == "includes":
            return right in left
        left = left.strip(string.whitespace + string.punctuation)
        right = right.strip(string.whitespace + string.punctuation)
        if self._config.location == "begin":
            return left.startswith(right)
        if self._config.location == "any":
            return right in left
        if self._config.location == "exact":
            return left == right
        return left.endswith(right)


class InspectPatternScorerConfig(BaseModel):
    """Strict configuration for native Inspect regular-expression scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_values: tuple[str, ...]
    pattern: str
    ignore_case: bool = True
    match_all: bool = False


class InspectPatternScorer:
    """Native execution of Inspect 0.3.233 ``pattern`` scorer semantics."""

    def __init__(
        self,
        *,
        config: InspectPatternScorerConfig,
        memory: MemoryInterface | None = None,
    ) -> None:
        """Initialize the scorer."""
        self._config = config
        self._memory = memory or CentralMemory.get_memory_instance()
        self._compiled = re.compile(config.pattern, re.IGNORECASE if config.ignore_case else 0)

    @classmethod
    def from_config(cls, config: Mapping[str, JSONValue]) -> InspectPatternScorer:
        """
        Build the scorer from strict manifest configuration.

        Returns:
            InspectPatternScorer: The configured native scorer.
        """
        return cls(config=InspectPatternScorerConfig.model_validate(dict(config)))

    async def score_result_async(self, *, result: CapabilityTaskResult, objective: str) -> list[Score]:
        """
        Score the final completion using captured regular-expression groups.

        Returns:
            list[Score]: The normalized score.
        """
        completion = _result_completion(result=result, memory=self._memory)
        match = self._compiled.search(completion)
        answer, value = self._match_groups(match.groups() if match else ())
        piece_id = str(result.final_message_piece_ids[-1]) if result.final_message_piece_ids else str(result.case_id)
        return normalize_inspect_score(
            score=InspectScore(
                value=value,
                answer=answer,
                explanation=completion if match else f"Scoring pattern not matched in output: {completion}",
            ),
            message_piece_id=piece_id,
            objective=objective,
        )

    def _match_groups(self, groups: tuple[str | None, ...]) -> tuple[str | None, str]:
        if not groups:
            return None, "N"
        targets = {
            target.lower() if self._config.ignore_case else target: target for target in self._config.expected_values
        }
        normalized = tuple(
            group.lower() if self._config.ignore_case and isinstance(group, str) else group for group in groups
        )
        if self._config.match_all:
            for group in normalized:
                if isinstance(group, str) and group not in targets:
                    return None, "I"
            # Preserve Inspect 0.3.233's Target.text behavior, including optional unmatched groups.
            answer = "".join(self._config.expected_values)
            return answer, "C" if answer else "I"
        found_index = next(
            (index for index, group in enumerate(normalized) if isinstance(group, str) and group in targets),
            None,
        )
        if found_index is not None:
            answer = groups[found_index]
            return answer, "C" if answer else "I"
        return (groups[0] if len(groups) == 1 else None), "I"


def _result_completion(*, result: CapabilityTaskResult, memory: MemoryInterface) -> str:
    pieces = memory.get_message_pieces(prompt_ids=list(result.final_message_piece_ids))
    by_id = {piece.id: piece for piece in pieces}
    return "\n".join(
        by_id[piece_id].converted_value
        for piece_id in result.final_message_piece_ids
        if piece_id in by_id and by_id[piece_id].converted_value
    )


def normalize_inspect_score(
    *,
    score: InspectScore,
    message_piece_id: str,
    objective: str,
) -> list[Score]:
    """
    Normalize scalar or dict-valued Inspect scores into canonical PyRIT scores.

    Returns:
        list[Score]: One canonical score per scalar or dictionary entry.

    Raises:
        ValueError: If a list-valued score cannot be executed by native aggregation.
    """
    values = score.value if isinstance(score.value, dict) else {None: score.value}
    normalized = []
    metadata: dict[str, str | int | float] = {}
    for metadata_key, metadata_value in score.metadata.items():
        if not isinstance(metadata_value, (str, int, float)):
            raise ValueError(
                "Inspect score metadata values must be strings or numbers for canonical PyRIT score transport."
            )
        metadata[str(metadata_key)] = metadata_value
    for key, value in values.items():
        if isinstance(value, list):
            raise ValueError("List-valued Inspect scores are represented but not executable in native aggregation.")
        score_type, score_value = _normalized_value(value)
        score_metadata = {
            **metadata,
            **({"score_key": key} if key is not None else {}),
            **({"answer": score.answer} if score.answer is not None else {}),
        }
        normalized.append(
            Score(
                score_value=score_value,
                score_type=score_type,
                score_category=[],
                score_rationale=score.explanation,
                score_metadata=score_metadata,
                message_piece_id=message_piece_id,
                objective=objective,
            )
        )
    return normalized


def _normalized_value(
    value: bool | int | float | str,
) -> tuple[Literal["true_false", "float_scale", "unknown"], str]:
    if isinstance(value, bool):
        return "true_false", str(value)
    if isinstance(value, (int, float)) and 0 <= value <= 1:
        return "float_scale", str(float(value))
    if isinstance(value, (int, float)):
        return "unknown", str(value)
    inspect_labels = {"C": "True", "I": "False", "P": "0.5", "N": "False"}
    if value in {"C", "I", "N"}:
        return "true_false", inspect_labels[value]
    if value == "P":
        return "float_scale", inspect_labels[value]
    return "unknown", value


class InspectCheckFlagConfig(BaseModel):
    """Strict configuration for the pinned in-house ``check_flag`` callback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: str = Field(min_length=1)
    command: str = Field(min_length=1)
    source_identity: str = Field(min_length=1)
    timeout_seconds: float = Field(default=300.0, gt=0, le=300.0)


class InspectSandboxCallbackProxy:
    """Typed, bounded scorer-only proxy to a live sandbox session."""

    def __init__(
        self,
        *,
        session: SandboxSession,
        allowed_environments: tuple[str, ...],
        timeout_seconds: float,
    ) -> None:
        """Initialize a callback proxy with an explicit environment allow-list."""
        self._session = session
        self._allowed_environments = frozenset(allowed_environments)
        self._timeout_seconds = timeout_seconds

    async def exec_async(
        self,
        *,
        environment: str,
        argv: tuple[str, ...],
        cancellation_event: asyncio.Event | None = None,
    ) -> tuple[bytes, bytes, bool]:
        """
        Execute one bounded callback command in an explicitly allowed environment.

        Returns:
            tuple[bytes, bytes, bool]: Standard output, standard error, and success.

        Raises:
            ValueError: If the environment or argv is outside the bounded callback contract.
        """
        if environment not in self._allowed_environments:
            raise ValueError(f"Compatibility callback environment '{environment}' is not allowed.")
        if not argv or len(argv) > 16 or any(len(item) > 16_384 for item in argv):
            raise ValueError("Compatibility callback argv exceeds the bounded RPC shape.")
        result = await self._session.get_environment(environment).exec_async(
            request=SandboxExecRequest(argv=argv, timeout_seconds=self._timeout_seconds),
            cancellation_event=cancellation_event,
        )
        success = result.status is SandboxOperationStatus.SUCCEEDED and result.exit_code == 0
        return result.stdout, result.stderr, success


class InspectCheckFlagScorer:
    """Native execution of the pinned in-house ``check_flag`` callback contract."""

    def __init__(
        self,
        *,
        config: InspectCheckFlagConfig,
        memory: MemoryInterface | None = None,
    ) -> None:
        """Initialize the exact pinned callback scorer."""
        self._config = config
        self._memory = memory or CentralMemory.get_memory_instance()

    @classmethod
    def from_config(cls, config: Mapping[str, JSONValue]) -> InspectCheckFlagScorer:
        """
        Build the scorer from a strict callback descriptor.

        Returns:
            InspectCheckFlagScorer: The validated scorer.
        """
        return cls(config=InspectCheckFlagConfig.model_validate(dict(config)))

    async def score_async(
        self,
        *,
        result: CapabilityTaskResult,
        objective: str,
        session: SandboxSession,
        cancellation_event: asyncio.Event | None = None,
    ) -> list[Score]:
        """
        Read the live flag and compare it with the preserved final answer.

        Returns:
            list[Score]: One native true/false score.
        """
        proxy = InspectSandboxCallbackProxy(
            session=session,
            allowed_environments=(self._config.environment,),
            timeout_seconds=self._config.timeout_seconds,
        )
        stdout, stderr, success = await proxy.exec_async(
            environment=self._config.environment,
            argv=("sh", "-c", self._config.command),
            cancellation_event=cancellation_event,
        )
        completion = self._completion(result)
        flag = stdout.decode("utf-8", errors="replace").strip()
        matched = success and bool(flag) and flag in completion
        explanation = (
            f"Flag: {flag}\n{completion}"
            if success
            else f"Error reading flag from container: {stderr.decode('utf-8', errors='replace')}"
        )
        piece_id = result.final_message_piece_ids[-1] if result.final_message_piece_ids else str(result.case_id)
        return [
            Score(
                score_value=str(matched),
                score_type="true_false",
                score_category=[],
                score_rationale=f"{self._config.source_identity}: {explanation}",
                message_piece_id=piece_id,
                objective=objective,
            )
        ]

    def _completion(self, result: CapabilityTaskResult) -> str:
        pieces = self._memory.get_message_pieces(prompt_ids=list(result.final_message_piece_ids))
        by_id = {piece.id: piece for piece in pieces}
        return "\n".join(
            by_id[piece_id].converted_value
            for piece_id in result.final_message_piece_ids
            if piece_id in by_id and by_id[piece_id].converted_value
        )
