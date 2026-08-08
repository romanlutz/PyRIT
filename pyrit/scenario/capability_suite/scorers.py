# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Native and session-aware scorers for capability-suite runs.

Scorers here only evaluate: they read a completed ``CapabilityTaskResult`` (and,
for session-aware scorers, a still-open ``SandboxSession``) and return ``Score``
objects. They never branch execution or retry logic — that remains the runner's
responsibility (see ``pyrit.scenario.capability_suite.runner``).

Two scorer seams are supported, unified behind one protocol the runner calls:

* ``CapabilitySuiteScorer`` -- the seam the runner invokes for every configured
  scorer, given the completed result, the case objective, and the still-open
  sandbox session (satisfies "score while sandbox alive").
* ``CapabilityResultScorer`` (from ``pyrit.executor.capability``) -- the narrower,
  session-agnostic seam already used by ``CapabilityTaskExecutor`` and by the
  existing ``MessageScorerAdapter``. ``ResultOnlyScorerAdapter`` below adapts any
  such scorer into the session-aware seam by simply ignoring the session, which is
  how an *existing* ``pyrit.score.Scorer`` (wrapped in ``MessageScorerAdapter``) is
  composed into a capability suite without reimplementing anything.
"""

from __future__ import annotations

import base64
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyrit.executor.capability import ToolExecutionEvidence, ToolExecutionStatus
from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import JSONValue, Score
from pyrit.sandbox import SandboxExecRequest, SandboxOperationStatus
from pyrit.scenario.capability_suite.manifest import validate_safe_relative_path

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyrit.executor.capability import CapabilityResultScorer, CapabilityTaskResult
    from pyrit.sandbox import SandboxSession


class CapabilitySuiteScorer(Protocol):
    """A scorer invoked by the runner while the attempt's sandbox session is alive."""

    async def score_async(
        self,
        *,
        result: CapabilityTaskResult,
        objective: str,
        session: SandboxSession,
    ) -> list[Score]:
        """Score a completed attempt, optionally using the still-open sandbox session."""


class ResultOnlyScorerAdapter:
    """Adapt a session-agnostic ``CapabilityResultScorer`` into the suite scoring seam."""

    def __init__(self, *, scorer: CapabilityResultScorer) -> None:
        """Initialize the adapter around an existing result scorer."""
        self._scorer = scorer

    async def score_async(
        self,
        *,
        result: CapabilityTaskResult,
        objective: str,
        session: SandboxSession,
    ) -> list[Score]:
        """
        Score using only the completed result, ignoring the sandbox session.

        Returns:
            list[Score]: Scores produced by the wrapped result scorer.
        """
        _ = session
        return await self._scorer.score_result_async(result=result, objective=objective)


class TextMatchMode(str, Enum):
    """How ``TextMatchScorer`` compares the final message text to an expected value."""

    EXACT = "exact"
    SUBSTRING = "substring"


class TextMatchScorerConfig(BaseModel):
    """Strict manifest configuration for ``TextMatchScorer``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_value: str
    mode: TextMatchMode = TextMatchMode.SUBSTRING
    case_sensitive: bool = False
    category: str | None = None


class ToolEvidenceScorerConfig(BaseModel):
    """Strict manifest configuration for ``ToolEvidenceScorer``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str = Field(min_length=1)
    expected_status: ToolExecutionStatus = ToolExecutionStatus.SUCCEEDED
    category: str | None = None


class SandboxFileScorerConfig(BaseModel):
    """Strict manifest configuration for ``SandboxFileScorer``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    environment: str | None = None
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_content_base64: str | None = None
    category: str | None = None

    @model_validator(mode="after")
    def _validate_path(self) -> SandboxFileScorerConfig:
        validate_safe_relative_path(self.path)
        return self


class SandboxCommandScorerConfig(BaseModel):
    """Strict manifest configuration for ``SandboxCommandScorer``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: tuple[str, ...] | None = None
    shell_script: str | None = None
    environment: str | None = None
    expected_exit_code: int = 0
    category: str | None = None

    @model_validator(mode="after")
    def _validate_command(self) -> SandboxCommandScorerConfig:
        if (self.argv is None) == (self.shell_script is None):
            raise ValueError("Exactly one of 'argv' or 'shell_script' must be provided.")
        if self.argv is not None and not self.argv:
            raise ValueError("'argv' must contain at least one element.")
        return self


class SandboxStateMatchMode(str, Enum):
    """How live sandbox command output is compared with the final model response."""

    EXACT = "exact"
    STATE_IN_RESPONSE = "state_in_response"
    RESPONSE_IN_STATE = "response_in_state"


class SandboxStateMatchScorerConfig(BaseModel):
    """Strict manifest configuration for ``SandboxStateMatchScorer``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: tuple[str, ...] | None = None
    shell_script: str | None = None
    environment: str | None = None
    expected_exit_code: int = 0
    mode: SandboxStateMatchMode = SandboxStateMatchMode.STATE_IN_RESPONSE
    case_sensitive: bool = True
    category: str | None = None

    @model_validator(mode="after")
    def _validate_command(self) -> SandboxStateMatchScorerConfig:
        if (self.argv is None) == (self.shell_script is None):
            raise ValueError("Exactly one of 'argv' or 'shell_script' must be provided.")
        if self.argv is not None and not self.argv:
            raise ValueError("'argv' must contain at least one element.")
        return self


def _final_message_text(*, result: CapabilityTaskResult, memory: MemoryInterface) -> str:
    pieces = memory.get_message_pieces(prompt_ids=list(result.final_message_piece_ids))
    by_id = {piece.id: piece for piece in pieces}
    ordered = [by_id[piece_id] for piece_id in result.final_message_piece_ids if piece_id in by_id]
    return "\n".join(piece.converted_value for piece in ordered if piece.converted_value)


class TextMatchScorer:
    """Native scorer that checks the final assistant text for an exact or substring match."""

    def __init__(
        self,
        *,
        expected_value: str,
        mode: TextMatchMode = TextMatchMode.SUBSTRING,
        case_sensitive: bool = False,
        category: str | None = None,
        memory: MemoryInterface | None = None,
    ) -> None:
        """Initialize the native text-match scorer."""
        self._expected_value = expected_value
        self._mode = mode
        self._case_sensitive = case_sensitive
        self._category = category
        self._memory = memory or CentralMemory.get_memory_instance()

    @classmethod
    def from_config(cls, config: Mapping[str, JSONValue]) -> TextMatchScorer:
        """
        Build a ``TextMatchScorer`` from a manifest scorer JSON config.

        Returns:
            TextMatchScorer: The configured scorer.

        Raises:
            ValueError: If the config is missing required fields or has invalid types.
        """
        parsed = TextMatchScorerConfig.model_validate(dict(config))
        return cls(
            expected_value=parsed.expected_value,
            mode=parsed.mode,
            case_sensitive=parsed.case_sensitive,
            category=parsed.category,
        )

    async def score_result_async(self, *, result: CapabilityTaskResult, objective: str) -> list[Score]:
        """
        Score the final assistant text against the configured expected value.

        Returns:
            list[Score]: A single ``true_false`` score.
        """
        text = _final_message_text(result=result, memory=self._memory)
        matched = self._is_match(text)
        piece_id = result.final_message_piece_ids[-1] if result.final_message_piece_ids else str(result.case_id)
        return [
            Score(
                score_value=str(matched),
                score_type="true_false",
                score_category=[self._category] if self._category else [],
                score_rationale=f"TextMatchScorer[{self._mode.value}] against final message text.",
                message_piece_id=piece_id,
                objective=objective,
            )
        ]

    def _is_match(self, text: str) -> bool:
        if self._mode is TextMatchMode.EXACT:
            left, right = self._expected_value.strip(), text.strip()
            if not self._case_sensitive:
                left, right = left.lower(), right.lower()
            return left == right
        left, right = self._expected_value, text
        if not self._case_sensitive:
            left, right = left.lower(), right.lower()
        return left in right


class ToolEvidenceScorer:
    """Native scorer that checks whether a named tool reached an expected status."""

    def __init__(
        self,
        *,
        tool_name: str,
        expected_status: ToolExecutionStatus = ToolExecutionStatus.SUCCEEDED,
        category: str | None = None,
    ) -> None:
        """Initialize the tool-evidence scorer."""
        self._tool_name = tool_name
        self._expected_status = expected_status
        self._category = category

    @classmethod
    def from_config(cls, config: Mapping[str, JSONValue]) -> ToolEvidenceScorer:
        """
        Build a ``ToolEvidenceScorer`` from a manifest scorer JSON config.

        Returns:
            ToolEvidenceScorer: The configured scorer.

        Raises:
            ValueError: If the config is missing required fields or has invalid types.
        """
        parsed = ToolEvidenceScorerConfig.model_validate(dict(config))
        return cls(
            tool_name=parsed.tool_name,
            expected_status=parsed.expected_status,
            category=parsed.category,
        )

    async def score_result_async(self, *, result: CapabilityTaskResult, objective: str) -> list[Score]:
        """
        Score whether the named tool reached the expected status at least once.

        Returns:
            list[Score]: A single ``true_false`` score.
        """
        matched = any(
            isinstance(evidence, ToolExecutionEvidence)
            and evidence.tool_name == self._tool_name
            and evidence.status is self._expected_status
            for evidence in result.evidence
        )
        return [
            Score(
                score_value=str(matched),
                score_type="true_false",
                score_category=[self._category] if self._category else [],
                score_rationale=f"ToolEvidenceScorer[{self._tool_name} == {self._expected_status.value}].",
                message_piece_id=str(result.case_id),
                objective=objective,
            )
        ]


class SandboxFileScorer:
    """Session-aware scorer that reads a live sandbox file and compares hash/content."""

    def __init__(
        self,
        *,
        path: str,
        environment: str | None = None,
        expected_sha256: str | None = None,
        expected_content: bytes | None = None,
        category: str | None = None,
    ) -> None:
        """Initialize the sandbox-file scorer."""
        self._path = path
        self._environment = environment
        self._expected_sha256 = expected_sha256
        self._expected_content = expected_content
        self._category = category

    @classmethod
    def from_config(cls, config: Mapping[str, JSONValue]) -> SandboxFileScorer:
        """
        Build a ``SandboxFileScorer`` from a manifest scorer JSON config.

        Returns:
            SandboxFileScorer: The configured scorer.

        Raises:
            ValueError: If the config is missing required fields or has invalid types.
        """
        parsed = SandboxFileScorerConfig.model_validate(dict(config))
        expected_content: bytes | None = None
        if parsed.expected_content_base64 is not None:
            expected_content = base64.b64decode(parsed.expected_content_base64, validate=True)
        return cls(
            path=parsed.path,
            environment=parsed.environment,
            expected_sha256=parsed.expected_sha256,
            expected_content=expected_content,
            category=parsed.category,
        )

    async def score_async(
        self,
        *,
        result: CapabilityTaskResult,
        objective: str,
        session: SandboxSession,
    ) -> list[Score]:
        """
        Read the configured file from the live sandbox and compare hash/content.

        Returns:
            list[Score]: A single ``true_false`` score.
        """
        environment = session.get_environment(self._environment)
        read_result = await environment.read_file_async(path=self._path)
        matched = read_result.status is SandboxOperationStatus.SUCCEEDED
        if matched and self._expected_sha256 is not None:
            matched = read_result.sha256 == self._expected_sha256
        if matched and self._expected_content is not None:
            matched = (read_result.data or b"") == self._expected_content
        return [
            Score(
                score_value=str(matched),
                score_type="true_false",
                score_category=[self._category] if self._category else [],
                score_rationale=f"SandboxFileScorer[{self._path}] status={read_result.status.value}.",
                message_piece_id=str(result.case_id),
                objective=objective,
            )
        ]


class SandboxCommandScorer:
    """Session-aware scorer that runs a live sandbox command and checks its exit code."""

    def __init__(
        self,
        *,
        argv: tuple[str, ...] | None = None,
        shell_script: str | None = None,
        environment: str | None = None,
        expected_exit_code: int = 0,
        category: str | None = None,
    ) -> None:
        """
        Initialize the sandbox-command scorer.

        Raises:
            ValueError: If neither or both of ``argv``/``shell_script`` are provided.
        """
        if (argv is None) == (shell_script is None):
            raise ValueError("Exactly one of 'argv' or 'shell_script' must be provided.")
        self._argv = argv
        self._shell_script = shell_script
        self._environment = environment
        self._expected_exit_code = expected_exit_code
        self._category = category

    @classmethod
    def from_config(cls, config: Mapping[str, JSONValue]) -> SandboxCommandScorer:
        """
        Build a ``SandboxCommandScorer`` from a manifest scorer JSON config.

        Returns:
            SandboxCommandScorer: The configured scorer.

        Raises:
            ValueError: If the config is missing required fields or has invalid types.
        """
        parsed = SandboxCommandScorerConfig.model_validate(dict(config))
        return cls(
            argv=parsed.argv,
            shell_script=parsed.shell_script,
            environment=parsed.environment,
            expected_exit_code=parsed.expected_exit_code,
            category=parsed.category,
        )

    async def score_async(
        self,
        *,
        result: CapabilityTaskResult,
        objective: str,
        session: SandboxSession,
    ) -> list[Score]:
        """
        Run the configured command in the live sandbox and compare its exit code.

        Returns:
            list[Score]: A single ``true_false`` score.
        """
        environment = session.get_environment(self._environment)
        exec_result = await environment.exec_async(
            request=SandboxExecRequest(argv=self._argv, shell_script=self._shell_script)
        )
        matched = (
            exec_result.status is SandboxOperationStatus.SUCCEEDED and exec_result.exit_code == self._expected_exit_code
        )
        return [
            Score(
                score_value=str(matched),
                score_type="true_false",
                score_category=[self._category] if self._category else [],
                score_rationale=(
                    f"SandboxCommandScorer status={exec_result.status.value} exit_code={exec_result.exit_code}."
                ),
                message_piece_id=str(result.case_id),
                objective=objective,
            )
        ]


class SandboxStateMatchScorer:
    """Compare final model text with dynamic state read from a live sandbox command."""

    def __init__(
        self,
        *,
        argv: tuple[str, ...] | None = None,
        shell_script: str | None = None,
        environment: str | None = None,
        expected_exit_code: int = 0,
        mode: SandboxStateMatchMode = SandboxStateMatchMode.STATE_IN_RESPONSE,
        case_sensitive: bool = True,
        category: str | None = None,
        memory: MemoryInterface | None = None,
    ) -> None:
        """
        Initialize the live-state matcher.

        Raises:
            ValueError: If neither or both of ``argv``/``shell_script`` are provided.
        """
        if (argv is None) == (shell_script is None):
            raise ValueError("Exactly one of 'argv' or 'shell_script' must be provided.")
        self._argv = argv
        self._shell_script = shell_script
        self._environment = environment
        self._expected_exit_code = expected_exit_code
        self._mode = mode
        self._case_sensitive = case_sensitive
        self._category = category
        self._memory = memory or CentralMemory.get_memory_instance()

    @classmethod
    def from_config(cls, config: Mapping[str, JSONValue]) -> SandboxStateMatchScorer:
        """
        Build a ``SandboxStateMatchScorer`` from manifest JSON config.

        Returns:
            SandboxStateMatchScorer: The configured scorer.
        """
        parsed = SandboxStateMatchScorerConfig.model_validate(dict(config))
        return cls(
            argv=parsed.argv,
            shell_script=parsed.shell_script,
            environment=parsed.environment,
            expected_exit_code=parsed.expected_exit_code,
            mode=parsed.mode,
            case_sensitive=parsed.case_sensitive,
            category=parsed.category,
        )

    async def score_async(
        self,
        *,
        result: CapabilityTaskResult,
        objective: str,
        session: SandboxSession,
    ) -> list[Score]:
        """
        Read sandbox state before cleanup and compare it with the submitted response.

        Returns:
            list[Score]: A single evidence-aware ``true_false`` score.
        """
        environment = session.get_environment(self._environment)
        exec_result = await environment.exec_async(
            request=SandboxExecRequest(argv=self._argv, shell_script=self._shell_script)
        )
        state = exec_result.stdout.decode("utf-8", errors="replace").strip()
        response = _final_message_text(result=result, memory=self._memory).strip()
        command_succeeded = (
            exec_result.status is SandboxOperationStatus.SUCCEEDED
            and exec_result.exit_code == self._expected_exit_code
            and bool(state)
        )
        matched = command_succeeded and self._is_match(state=state, response=response)
        return [
            Score(
                score_value=str(matched),
                score_type="true_false",
                score_category=[self._category] if self._category else [],
                score_rationale=(
                    "SandboxStateMatchScorer compared final response with live sandbox state; "
                    f"status={exec_result.status.value} exit_code={exec_result.exit_code}."
                ),
                message_piece_id=(
                    result.final_message_piece_ids[-1] if result.final_message_piece_ids else str(result.case_id)
                ),
                objective=objective,
            )
        ]

    def _is_match(self, *, state: str, response: str) -> bool:
        if not self._case_sensitive:
            state, response = state.lower(), response.lower()
        if self._mode is SandboxStateMatchMode.EXACT:
            return state == response
        if self._mode is SandboxStateMatchMode.RESPONSE_IN_STATE:
            return bool(response) and response in state
        return state in response
