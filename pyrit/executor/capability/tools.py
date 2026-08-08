# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Provider-neutral tool declaration, registration, and execution."""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from pyrit.executor.capability.models import (
    ApprovalDecisionKind,
    ApprovalEvidence,
    ArtifactEvidence,
    CapabilityEvidence,
    ErrorEvidence,
    ToolExecutionEvidence,
    ToolExecutionStatus,
)
from pyrit.models import JSONValue, ToolCallError, ToolCallRequest, ToolCallResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping


def _default_input_schema() -> dict[str, JSONValue]:
    """Return the default object schema for a tool declaration."""
    return {"type": "object", "properties": {}}


class ToolDeclaration(BaseModel):
    """A model-visible tool declaration, separate from its implementation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = None
    input_schema: dict[str, JSONValue] = Field(default_factory=_default_input_schema)
    idempotent: bool = False
    max_retries: int = Field(default=0, ge=0)
    timeout_seconds: float | None = Field(default=None, gt=0)
    retryable_error_codes: tuple[str, ...] = ()


class ToolArtifact(BaseModel):
    """An artifact reference returned by a tool implementation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: str
    uri: str
    media_type: str | None = None
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class ToolExecutionOutput(BaseModel):
    """A successful tool implementation result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    output: JSONValue
    artifacts: tuple[ToolArtifact, ...] = ()
    evidence: tuple[CapabilityEvidence, ...] = ()
    side_effect_completed: bool = False
    completion_answer: str | None = None
    continuation_message: str | None = None


class ToolExecutionContext(BaseModel):
    """Stable context supplied to every tool implementation attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    case_id: uuid.UUID
    conversation_id: str
    call_id: str
    attempt_id: uuid.UUID
    attempt_number: int = Field(gt=0)
    asset_references: tuple[str, ...] = ()
    environment_requirement_references: tuple[str, ...] = ()
    cancellation_event: asyncio.Event | None = None


class ToolImplementation(Protocol):
    """An asynchronous provider-neutral tool implementation."""

    async def execute_async(
        self,
        *,
        arguments: Mapping[str, JSONValue],
        context: ToolExecutionContext,
    ) -> ToolExecutionOutput:
        """Execute one tool attempt."""


class ToolExecutionError(Exception):
    """A declared tool failure with retry and side-effect facts."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool = False,
        side_effect_completed: bool | None = False,
        details: JSONValue = None,
        status: ToolExecutionStatus = ToolExecutionStatus.FAILED,
        evidence: tuple[CapabilityEvidence, ...] = (),
    ) -> None:
        """Initialize a declared tool failure."""
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.side_effect_completed = side_effect_completed
        self.details = details
        self.status = status
        self.evidence = evidence


class CooperativeCancellationError(Exception):
    """Cooperative cancellation requested through an execution context."""


class ToolApprovalDecision(BaseModel):
    """A policy decision returned before any tool call is executed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: ApprovalDecisionKind
    policy: str
    reason: str | None = None


class ToolApprovalPolicy(Protocol):
    """A policy that approves or denies a validated tool call."""

    async def decide_async(
        self,
        *,
        request: ToolCallRequest,
        declaration: ToolDeclaration,
        arguments: Mapping[str, JSONValue],
        context: ToolExecutionContext,
    ) -> ToolApprovalDecision:
        """Return the explicit decision for one call."""


class AllowAllToolApprovalPolicy:
    """An explicit allow policy suitable when no external approval is required."""

    async def decide_async(
        self,
        *,
        request: ToolCallRequest,
        declaration: ToolDeclaration,
        arguments: Mapping[str, JSONValue],
        context: ToolExecutionContext,
    ) -> ToolApprovalDecision:
        """
        Allow the validated call.

        Returns:
            ToolApprovalDecision: An explicit allow decision.
        """
        return ToolApprovalDecision(decision=ApprovalDecisionKind.ALLOW, policy=type(self).__name__)


@dataclass(frozen=True)
class RegisteredTool:
    """A declaration bound to an implementation."""

    declaration: ToolDeclaration
    implementation: ToolImplementation


class ToolRegistry:
    """A registry that keeps tool declarations separate from implementations."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._declarations: dict[str, ToolDeclaration] = {}
        self._implementations: dict[str, ToolImplementation] = {}

    def register_declaration(self, declaration: ToolDeclaration) -> None:
        """
        Register one model-visible declaration.

        Raises:
            ValueError: If the declaration name is already registered.
        """
        if declaration.name in self._declarations:
            raise ValueError(f"Tool declaration '{declaration.name}' is already registered.")
        self._declarations[declaration.name] = declaration

    def register_implementation(self, *, name: str, implementation: ToolImplementation) -> None:
        """
        Register one implementation independently of its declaration.

        Raises:
            ValueError: If the implementation name is already registered.
        """
        if name in self._implementations:
            raise ValueError(f"Tool implementation '{name}' is already registered.")
        self._implementations[name] = implementation

    def register(self, *, declaration: ToolDeclaration, implementation: ToolImplementation) -> None:
        """
        Register a declaration and implementation together.

        Raises:
            ValueError: If either side is already registered.
        """
        self.register_declaration(declaration)
        try:
            self.register_implementation(name=declaration.name, implementation=implementation)
        except ValueError:
            self._declarations.pop(declaration.name, None)
            raise

    def get(self, name: str) -> RegisteredTool | None:
        """Return a complete binding, or None when either side is missing."""
        declaration = self._declarations.get(name)
        implementation = self._implementations.get(name)
        if declaration is None or implementation is None:
            return None
        return RegisteredTool(declaration=declaration, implementation=implementation)

    def declarations(self, *, names: tuple[str, ...] | None = None) -> tuple[ToolDeclaration, ...]:
        """
        Return declarations in deterministic registration or requested order.

        Returns:
            tuple[ToolDeclaration, ...]: The requested declarations.

        Raises:
            ValueError: If a requested declaration is missing.
        """
        if names is None:
            return tuple(self._declarations.values())
        missing = [name for name in names if name not in self._declarations]
        if missing:
            raise ValueError(f"Required tool declarations are missing: {', '.join(missing)}")
        return tuple(self._declarations[name] for name in names)


@dataclass(frozen=True)
class PreparedToolCall:
    """A validated call and its explicit policy decision."""

    request: ToolCallRequest
    request_piece_id: uuid.UUID
    binding: RegisteredTool | None = None
    arguments: dict[str, JSONValue] | None = None
    validation_error: ToolCallError | None = None
    approval: ToolApprovalDecision | None = None

    @property
    def denied(self) -> bool:
        """Whether policy denied this call."""
        return self.approval is not None and self.approval.decision is ApprovalDecisionKind.DENY


@dataclass(frozen=True)
class ToolExecutionRecord:
    """A canonical result plus authoritative evidence."""

    result: ToolCallResult
    execution_evidence: tuple[ToolExecutionEvidence, ...] = ()
    error_evidence: tuple[ErrorEvidence, ...] = ()
    artifact_evidence: tuple[ArtifactEvidence, ...] = ()
    additional_evidence: tuple[CapabilityEvidence, ...] = ()
    completion_answer: str | None = None
    continuation_message: str | None = None


class CapabilityToolRuntime:
    """Validate, approve, and execute canonical tool requests."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        approval_policy: ToolApprovalPolicy | None = None,
    ) -> None:
        """Initialize the runtime."""
        self._registry = registry
        self._approval_policy = approval_policy or AllowAllToolApprovalPolicy()

    async def prepare_calls_async(
        self,
        *,
        calls: tuple[tuple[ToolCallRequest, uuid.UUID], ...],
        case_id: uuid.UUID,
        conversation_id: str,
        asset_references: tuple[str, ...],
        environment_requirement_references: tuple[str, ...],
        cancellation_event: asyncio.Event | None,
    ) -> tuple[tuple[PreparedToolCall, ...], tuple[ApprovalEvidence, ...]]:
        """
        Validate and approve every call before execution begins.

        Returns:
            tuple[tuple[PreparedToolCall, ...], tuple[ApprovalEvidence, ...]]:
                Prepared calls and policy evidence in request order.
        """
        prepared: list[PreparedToolCall] = []
        approvals: list[ApprovalEvidence] = []
        for request, piece_id in calls:
            item = await self._prepare_call_async(
                request=request,
                request_piece_id=piece_id,
                case_id=case_id,
                conversation_id=conversation_id,
                asset_references=asset_references,
                environment_requirement_references=environment_requirement_references,
                cancellation_event=cancellation_event,
            )
            prepared.append(item)
            if item.approval is not None:
                approvals.append(
                    ApprovalEvidence(
                        call_id=request.call_id,
                        request_piece_id=piece_id,
                        tool_name=request.name,
                        decision=item.approval.decision,
                        policy=item.approval.policy,
                        reason=item.approval.reason,
                    )
                )
        return tuple(prepared), tuple(approvals)

    async def execute_call_async(
        self,
        *,
        call: PreparedToolCall,
        case_id: uuid.UUID,
        conversation_id: str,
        asset_references: tuple[str, ...],
        environment_requirement_references: tuple[str, ...],
        cancellation_event: asyncio.Event | None,
    ) -> ToolExecutionRecord:
        """
        Execute one prepared call or return its deterministic non-execution result.

        Returns:
            ToolExecutionRecord: The canonical result and execution evidence.

        Raises:
            RuntimeError: If a supposedly validated call is incomplete.
        """
        if call.validation_error is not None:
            attempt_id = uuid.uuid4()
            now = datetime.now(tz=timezone.utc)
            evidence = ToolExecutionEvidence(
                call_id=call.request.call_id,
                request_piece_id=call.request_piece_id,
                attempt_id=attempt_id,
                attempt_number=1,
                tool_name=call.request.name,
                status=ToolExecutionStatus.NOT_EXECUTED,
                started_at=now,
                ended_at=now,
                error_code=call.validation_error.code,
            )
            error_evidence = ErrorEvidence(
                phase="tool_validation",
                code=call.validation_error.code,
                message=call.validation_error.message,
                call_id=call.request.call_id,
                attempt_id=attempt_id,
            )
            return ToolExecutionRecord(
                result=ToolCallResult(
                    call_id=call.request.call_id,
                    output=call.validation_error.message,
                    error=call.validation_error,
                ),
                execution_evidence=(evidence,),
                error_evidence=(error_evidence,),
            )
        if call.denied:
            approval = call.approval
            if approval is None:
                raise RuntimeError("Denied tool call is missing its approval decision.")
            error = ToolCallError(code="approval_denied", message=approval.reason or "Tool call denied by policy.")
            return self.not_executed_record(call=call, error=error, phase="tool_approval")
        if call.binding is None or call.arguments is None:
            raise RuntimeError("Prepared tool call is missing its validated binding or arguments.")
        return await self._execute_with_retries_async(
            call=call,
            case_id=case_id,
            conversation_id=conversation_id,
            asset_references=asset_references,
            environment_requirement_references=environment_requirement_references,
            cancellation_event=cancellation_event,
        )

    def not_executed_record(
        self,
        *,
        call: PreparedToolCall,
        error: ToolCallError,
        phase: str,
    ) -> ToolExecutionRecord:
        """
        Build a deterministic result for a call intentionally not executed.

        Returns:
            ToolExecutionRecord: The canonical non-execution result and evidence.
        """
        attempt_id = uuid.uuid4()
        now = datetime.now(tz=timezone.utc)
        evidence = ToolExecutionEvidence(
            call_id=call.request.call_id,
            request_piece_id=call.request_piece_id,
            attempt_id=attempt_id,
            attempt_number=1,
            tool_name=call.request.name,
            status=ToolExecutionStatus.NOT_EXECUTED,
            started_at=now,
            ended_at=now,
            error_code=error.code,
        )
        error_evidence = ErrorEvidence(
            phase=phase,
            code=error.code,
            message=error.message,
            call_id=call.request.call_id,
            attempt_id=attempt_id,
        )
        return ToolExecutionRecord(
            result=ToolCallResult(call_id=call.request.call_id, output=error.message, error=error),
            execution_evidence=(evidence,),
            error_evidence=(error_evidence,),
        )

    async def _prepare_call_async(
        self,
        *,
        request: ToolCallRequest,
        request_piece_id: uuid.UUID,
        case_id: uuid.UUID,
        conversation_id: str,
        asset_references: tuple[str, ...],
        environment_requirement_references: tuple[str, ...],
        cancellation_event: asyncio.Event | None,
    ) -> PreparedToolCall:
        binding = self._registry.get(request.name)
        if binding is None:
            error = ToolCallError(code="missing_tool", message=f"Tool '{request.name}' is not registered.")
            return PreparedToolCall(request=request, request_piece_id=request_piece_id, validation_error=error)
        try:
            arguments = json.loads(request.arguments)
        except json.JSONDecodeError:
            error = ToolCallError(
                code="malformed_arguments",
                message=f"Arguments for tool '{request.name}' are not valid JSON.",
                details={"raw_arguments": request.arguments},
            )
            return PreparedToolCall(request=request, request_piece_id=request_piece_id, validation_error=error)
        if not isinstance(arguments, dict):
            error = ToolCallError(
                code="malformed_arguments",
                message=f"Arguments for tool '{request.name}' must be a JSON object.",
            )
            return PreparedToolCall(request=request, request_piece_id=request_piece_id, validation_error=error)
        context = self._context(
            case_id=case_id,
            conversation_id=conversation_id,
            call_id=request.call_id,
            attempt_number=1,
            asset_references=asset_references,
            environment_requirement_references=environment_requirement_references,
            cancellation_event=cancellation_event,
        )
        approval = await self._approval_policy.decide_async(
            request=request,
            declaration=binding.declaration,
            arguments=arguments,
            context=context,
        )
        return PreparedToolCall(
            request=request,
            request_piece_id=request_piece_id,
            binding=binding,
            arguments=arguments,
            approval=approval,
        )

    async def _execute_with_retries_async(
        self,
        *,
        call: PreparedToolCall,
        case_id: uuid.UUID,
        conversation_id: str,
        asset_references: tuple[str, ...],
        environment_requirement_references: tuple[str, ...],
        cancellation_event: asyncio.Event | None,
    ) -> ToolExecutionRecord:
        if call.binding is None or call.arguments is None:
            raise RuntimeError("Validated tool call is incomplete.")
        execution_evidence: list[ToolExecutionEvidence] = []
        errors: list[ErrorEvidence] = []
        additional_evidence: list[CapabilityEvidence] = []
        max_attempts = 1 + (call.binding.declaration.max_retries if call.binding.declaration.idempotent else 0)
        for attempt_number in range(1, max_attempts + 1):
            context = self._context(
                case_id=case_id,
                conversation_id=conversation_id,
                call_id=call.request.call_id,
                attempt_number=attempt_number,
                asset_references=asset_references,
                environment_requirement_references=environment_requirement_references,
                cancellation_event=cancellation_event,
            )
            record = await self._execute_attempt_async(call=call, context=context)
            execution_evidence.append(record.execution_evidence[0])
            errors.extend(record.error_evidence)
            additional_evidence.extend(record.additional_evidence)
            if record.result.error is None:
                return replace(
                    record,
                    execution_evidence=tuple(execution_evidence),
                    error_evidence=tuple(errors),
                    additional_evidence=tuple(additional_evidence),
                )
            evidence = record.execution_evidence[0]
            should_retry = self._should_retry(
                call=call,
                evidence=evidence,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
            )
            if not should_retry:
                return replace(
                    record,
                    execution_evidence=tuple(execution_evidence),
                    error_evidence=tuple(errors),
                    additional_evidence=tuple(additional_evidence),
                )
        raise RuntimeError("Tool retry loop ended without a result.")

    async def _execute_attempt_async(
        self,
        *,
        call: PreparedToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionRecord:
        if call.binding is None or call.arguments is None:
            raise RuntimeError("Validated tool call is incomplete.")
        if context.cancellation_event is not None and context.cancellation_event.is_set():
            raise CooperativeCancellationError
        started_at = datetime.now(tz=timezone.utc)
        try:
            execution = call.binding.implementation.execute_async(arguments=call.arguments, context=context)
            output = await self._await_execution_async(
                execution=execution,
                timeout_seconds=call.binding.declaration.timeout_seconds,
                cancellation_event=context.cancellation_event,
            )
        except asyncio.TimeoutError:
            return self._error_record(
                call=call,
                context=context,
                started_at=started_at,
                status=ToolExecutionStatus.TIMED_OUT,
                error=ToolExecutionError(code="timeout", message="Tool execution timed out.", retryable=True),
            )
        except ToolExecutionError as error:
            return self._error_record(
                call=call,
                context=context,
                started_at=started_at,
                status=error.status,
                error=error,
            )
        except CooperativeCancellationError:
            raise
        except Exception as error:
            return self._error_record(
                call=call,
                context=context,
                started_at=started_at,
                status=ToolExecutionStatus.FAILED,
                error=ToolExecutionError(
                    code="implementation_error",
                    message=str(error),
                    side_effect_completed=None,
                    details={"exception_type": type(error).__name__},
                ),
            )
        result = ToolCallResult(call_id=call.request.call_id, output=self._serialize_output(output.output))
        ended_at = datetime.now(tz=timezone.utc)
        evidence = ToolExecutionEvidence(
            call_id=call.request.call_id,
            request_piece_id=call.request_piece_id,
            attempt_id=context.attempt_id,
            attempt_number=context.attempt_number,
            tool_name=call.request.name,
            status=ToolExecutionStatus.SUCCEEDED,
            started_at=started_at,
            ended_at=ended_at,
            side_effect_completed=output.side_effect_completed,
        )
        artifacts = tuple(
            ArtifactEvidence(
                artifact_id=artifact.artifact_id,
                uri=artifact.uri,
                media_type=artifact.media_type,
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
                created_by_call_id=call.request.call_id,
                metadata=artifact.metadata,
            )
            for artifact in output.artifacts
        )
        return ToolExecutionRecord(
            result=result,
            execution_evidence=(evidence,),
            artifact_evidence=artifacts,
            additional_evidence=output.evidence,
            completion_answer=output.completion_answer,
            continuation_message=output.continuation_message,
        )

    @staticmethod
    async def _await_execution_async(
        *,
        execution: Awaitable[ToolExecutionOutput],
        timeout_seconds: float | None,
        cancellation_event: asyncio.Event | None,
    ) -> ToolExecutionOutput:
        execution_task = asyncio.ensure_future(execution)
        if cancellation_event is None:
            return await asyncio.wait_for(execution_task, timeout=timeout_seconds)
        cancellation_task = asyncio.create_task(cancellation_event.wait())
        try:
            done, _ = await asyncio.wait(
                (execution_task, cancellation_task),
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            execution_task.cancel()
            cancellation_task.cancel()
            with suppress(asyncio.CancelledError):
                await execution_task
            with suppress(asyncio.CancelledError):
                await cancellation_task
            raise
        if execution_task in done:
            cancellation_task.cancel()
            with suppress(asyncio.CancelledError):
                await cancellation_task
            return execution_task.result()
        execution_task.cancel()
        with suppress(asyncio.CancelledError):
            await execution_task
        if execution_task.done() and not execution_task.cancelled() and execution_task.exception() is None:
            cancellation_task.cancel()
            with suppress(asyncio.CancelledError):
                await cancellation_task
            return execution_task.result()
        cancellation_task.cancel()
        with suppress(asyncio.CancelledError):
            await cancellation_task
        if cancellation_event.is_set():
            raise CooperativeCancellationError
        raise asyncio.TimeoutError

    def _error_record(
        self,
        *,
        call: PreparedToolCall,
        context: ToolExecutionContext,
        started_at: datetime,
        status: ToolExecutionStatus,
        error: ToolExecutionError,
    ) -> ToolExecutionRecord:
        ended_at = datetime.now(tz=timezone.utc)
        tool_error = ToolCallError(code=error.code, message=error.message, details=error.details)
        evidence = ToolExecutionEvidence(
            call_id=call.request.call_id,
            request_piece_id=call.request_piece_id,
            attempt_id=context.attempt_id,
            attempt_number=context.attempt_number,
            tool_name=call.request.name,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            side_effect_completed=error.side_effect_completed,
            error_code=error.code,
            retryable=error.retryable,
        )
        error_evidence = ErrorEvidence(
            phase="tool_execution",
            code=error.code,
            message=error.message,
            call_id=call.request.call_id,
            attempt_id=context.attempt_id,
        )
        return ToolExecutionRecord(
            result=ToolCallResult(call_id=call.request.call_id, output=error.message, error=tool_error),
            execution_evidence=(evidence,),
            error_evidence=(error_evidence,),
            additional_evidence=error.evidence,
        )

    @staticmethod
    def _should_retry(
        *,
        call: PreparedToolCall,
        evidence: ToolExecutionEvidence,
        attempt_number: int,
        max_attempts: int,
    ) -> bool:
        if call.binding is None:
            return False
        declaration = call.binding.declaration
        return (
            declaration.idempotent
            and attempt_number < max_attempts
            and not evidence.side_effect_completed
            and evidence.retryable
            and evidence.error_code in declaration.retryable_error_codes
        )

    @staticmethod
    def _context(
        *,
        case_id: uuid.UUID,
        conversation_id: str,
        call_id: str,
        attempt_number: int,
        asset_references: tuple[str, ...],
        environment_requirement_references: tuple[str, ...],
        cancellation_event: asyncio.Event | None,
    ) -> ToolExecutionContext:
        return ToolExecutionContext(
            case_id=case_id,
            conversation_id=conversation_id,
            call_id=call_id,
            attempt_id=uuid.uuid4(),
            attempt_number=attempt_number,
            asset_references=asset_references,
            environment_requirement_references=environment_requirement_references,
            cancellation_event=cancellation_event,
        )

    @staticmethod
    def _serialize_output(output: JSONValue) -> str:
        if isinstance(output, str):
            return output
        return json.dumps(output, sort_keys=True, separators=(",", ":"))
