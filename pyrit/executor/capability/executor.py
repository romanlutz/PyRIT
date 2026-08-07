# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Provider-neutral capability-task orchestration."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, TypeVar

from pyrit.executor.capability.models import (
    CapabilityCase,
    CapabilityEvidence,
    CapabilityOutcome,
    CapabilityTask,
    CapabilityTaskResult,
    CapabilityTerminationReason,
    CapabilityUsage,
    ErrorEvidence,
    LifecycleEventKind,
    LifecycleEvidence,
    ToolExecutionPolicy,
)
from pyrit.executor.capability.tools import (
    CapabilityToolRuntime,
    CooperativeCancellationError,
    PreparedToolCall,
    ToolExecutionRecord,
    ToolRegistry,
)
from pyrit.models import (
    Message,
    MessagePiece,
    TargetIdentifier,
    TargetInvocation,
    TokenUsage,
    ToolCallError,
    ToolCallRequest,
    ToolCallResult,
)
from pyrit.prompt_normalizer import PromptNormalizer

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from pyrit.executor.capability.scoring import CapabilityResultScorer
    from pyrit.executor.capability.target_adapter import CapabilityRequestOptionsFactory
    from pyrit.executor.capability.tools import ToolDeclaration
    from pyrit.prompt_target import PromptTarget

_AwaitedT = TypeVar("_AwaitedT")


class _WallClockLimitError(Exception):
    """The case wall-clock budget expired during an awaited operation."""


@dataclass
class _ExecutionState:
    """Mutable state kept private while constructing an immutable result."""

    started_at: datetime
    message_piece_ids: list[uuid.UUID] = field(default_factory=list)
    final_message_piece_ids: list[uuid.UUID] = field(default_factory=list)
    evidence: list[CapabilityEvidence] = field(default_factory=list)
    usage: CapabilityUsage = field(default_factory=CapabilityUsage)
    model_generations: int = 0
    tool_calls: int = 0
    consecutive_no_progress: int = 0
    consecutive_errors: int = 0
    pending_calls: tuple[PreparedToolCall, ...] = ()
    partial_tool_records: dict[int, ToolExecutionRecord] = field(default_factory=dict)


class CapabilityTaskExecutor:
    """Drive a capability conversation while keeping tool execution outside targets."""

    def __init__(
        self,
        *,
        target: PromptTarget,
        tool_registry: ToolRegistry,
        request_options_factory: CapabilityRequestOptionsFactory,
        tool_runtime: CapabilityToolRuntime | None = None,
        normalizer: PromptNormalizer | None = None,
        execution_policy: ToolExecutionPolicy = ToolExecutionPolicy.SEQUENTIAL,
        scorers: tuple[CapabilityResultScorer, ...] = (),
    ) -> None:
        """Initialize the executor and its injected seams."""
        self._target = target
        self._registry = tool_registry
        self._request_options_factory = request_options_factory
        self._tool_runtime = tool_runtime or CapabilityToolRuntime(registry=tool_registry)
        self._normalizer = normalizer or PromptNormalizer()
        self._execution_policy = execution_policy
        self._scorers = scorers

    def create_case(self, task: CapabilityTask) -> CapabilityCase:
        """
        Bind a task to this executor's target.

        Returns:
            CapabilityCase: A new immutable case.
        """
        target_identifier = TargetIdentifier.from_component_identifier(self._target.get_identifier())
        return CapabilityCase(task=task, target_identifier=target_identifier)

    async def execute_task_async(
        self,
        *,
        task: CapabilityTask,
        conversation_id: str | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> CapabilityTaskResult:
        """
        Create and execute one capability case.

        Returns:
            CapabilityTaskResult: The terminal result and evidence.

        Raises:
            asyncio.CancelledError: If the surrounding asyncio task is cancelled.
        """
        return await self.execute_case_async(
            case=self.create_case(task),
            conversation_id=conversation_id,
            cancellation_event=cancellation_event,
        )

    async def execute_case_async(
        self,
        *,
        case: CapabilityCase,
        conversation_id: str | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> CapabilityTaskResult:
        """
        Execute a case until completion, denial, cancellation, failure, or a limit.

        Returns:
            CapabilityTaskResult: The terminal result and evidence.

        Raises:
            asyncio.CancelledError: If the surrounding asyncio task is cancelled.
        """
        self._validate_case(case=case)
        conversation_id = conversation_id or str(uuid.uuid4())
        state = _ExecutionState(started_at=datetime.now(tz=timezone.utc))
        declarations = self._registry.declarations(names=case.task.required_tools or None)
        initial_messages = case.task.initial_messages or (Message.from_prompt(prompt=case.task.objective, role="user"),)
        try:
            current = await self._within_wall_clock_async(
                operation=self._persist_initial_messages_async(
                    messages=initial_messages,
                    conversation_id=conversation_id,
                    state=state,
                ),
                case=case,
                state=state,
            )
            current_persisted = False
            while True:
                termination = self._pre_generation_limit(case=case, state=state)
                if termination is not None:
                    outcome, reason, detail = termination
                    return await self._finish_async(
                        case=case,
                        conversation_id=conversation_id,
                        state=state,
                        outcome=outcome,
                        reason=reason,
                        detail=detail,
                    )
                self._raise_if_cancelled(cancellation_event=cancellation_event)
                if not current_persisted:
                    state.message_piece_ids.extend(piece.id for piece in current.message_pieces)
                response = await self._within_wall_clock_async(
                    operation=self._send_generation_async(
                        message=current,
                        conversation_id=conversation_id,
                        declarations=declarations,
                        request_already_persisted=current_persisted,
                    ),
                    case=case,
                    state=state,
                )
                state.model_generations += 1
                state.message_piece_ids.extend(piece.id for piece in response.message_pieces)
                self._record_generation_metadata(
                    request_piece_id=current.get_piece().id,
                    response=response,
                    state=state,
                )
                terminal = self._response_termination(response=response)
                usage_limit = self._usage_limit(case=case, state=state)
                calls = self._extract_tool_calls(response=response)
                tool_limit = self._tool_call_limit(case=case, state=state, additional_calls=len(calls))
                state.tool_calls += len(calls)
                if usage_limit is not None:
                    if calls:
                        result_message = self._error_results_message(
                            calls=calls,
                            code="usage_limit",
                            message=usage_limit,
                        )
                        await self._persist_result_message_async(
                            message=result_message,
                            conversation_id=conversation_id,
                            state=state,
                        )
                    return await self._finish_async(
                        case=case,
                        conversation_id=conversation_id,
                        state=state,
                        outcome=CapabilityOutcome.INCOMPLETE,
                        reason=CapabilityTerminationReason.LIMIT,
                        detail=usage_limit,
                    )
                if terminal is not None and calls:
                    outcome, reason, detail = terminal
                    result_message = self._error_results_message(
                        calls=calls,
                        code="generation_incomplete",
                        message=detail or "Target generation did not complete.",
                    )
                    await self._within_wall_clock_async(
                        operation=self._persist_result_message_async(
                            message=result_message,
                            conversation_id=conversation_id,
                            state=state,
                        ),
                        case=case,
                        state=state,
                    )
                    return await self._finish_async(
                        case=case,
                        conversation_id=conversation_id,
                        state=state,
                        outcome=outcome,
                        reason=reason,
                        detail=detail,
                    )
                if not calls:
                    outcome, reason, detail = terminal or (
                        CapabilityOutcome.COMPLETED,
                        CapabilityTerminationReason.COMPLETION,
                        None,
                    )
                    if outcome is CapabilityOutcome.COMPLETED:
                        state.final_message_piece_ids = [piece.id for piece in response.message_pieces]
                    return await self._finish_async(
                        case=case,
                        conversation_id=conversation_id,
                        state=state,
                        outcome=outcome,
                        reason=reason,
                        detail=detail,
                    )
                if tool_limit is not None:
                    result_message = self._error_results_message(
                        calls=calls,
                        code="tool_call_limit",
                        message=tool_limit,
                    )
                    await self._within_wall_clock_async(
                        operation=self._persist_result_message_async(
                            message=result_message,
                            conversation_id=conversation_id,
                            state=state,
                        ),
                        case=case,
                        state=state,
                    )
                    return await self._finish_async(
                        case=case,
                        conversation_id=conversation_id,
                        state=state,
                        outcome=CapabilityOutcome.INCOMPLETE,
                        reason=CapabilityTerminationReason.LIMIT,
                        detail=tool_limit,
                    )
                prepared, approvals = await self._within_wall_clock_async(
                    operation=self._tool_runtime.prepare_calls_async(
                        calls=calls,
                        case_id=case.case_id,
                        conversation_id=conversation_id,
                        asset_references=case.task.asset_references,
                        environment_requirement_references=case.task.environment_requirement_references,
                        cancellation_event=cancellation_event,
                    ),
                    case=case,
                    state=state,
                )
                state.evidence.extend(approvals)
                state.pending_calls = prepared
                state.partial_tool_records.clear()
                denied = any(call.denied for call in prepared)
                if denied:
                    records = self._deny_batch(calls=prepared)
                else:
                    records = await self._within_wall_clock_async(
                        operation=self._execute_calls_async(
                            calls=prepared,
                            case=case,
                            conversation_id=conversation_id,
                            cancellation_event=cancellation_event,
                            state=state,
                        ),
                        case=case,
                        state=state,
                    )
                result_message = self._result_message(records=records, state=state)
                await self._within_wall_clock_async(
                    operation=self._persist_result_message_async(
                        message=result_message,
                        conversation_id=conversation_id,
                        state=state,
                    ),
                    case=case,
                    state=state,
                )
                state.pending_calls = ()
                state.partial_tool_records.clear()
                self._update_progress(records=records, state=state)
                consecutive_limit = self._consecutive_limit(case=case, state=state)
                if denied or consecutive_limit is not None:
                    return await self._finish_async(
                        case=case,
                        conversation_id=conversation_id,
                        state=state,
                        outcome=CapabilityOutcome.DENIED if denied else CapabilityOutcome.INCOMPLETE,
                        reason=(CapabilityTerminationReason.DENIAL if denied else CapabilityTerminationReason.LIMIT),
                        detail="A tool call was denied by policy." if denied else consecutive_limit,
                    )
                current = result_message
                current_persisted = True
        except _WallClockLimitError:
            await self._persist_interrupted_tool_results_async(
                conversation_id=conversation_id,
                state=state,
                code="wall_clock_limit",
                message=f"Wall-clock limit of {case.task.limits.max_wall_clock_seconds} seconds reached.",
            )
            detail = f"Wall-clock limit of {case.task.limits.max_wall_clock_seconds} seconds reached."
            return await self._finish_async(
                case=case,
                conversation_id=conversation_id,
                state=state,
                outcome=CapabilityOutcome.INCOMPLETE,
                reason=CapabilityTerminationReason.LIMIT,
                detail=detail,
            )
        except CooperativeCancellationError:
            await self._persist_interrupted_tool_results_async(
                conversation_id=conversation_id,
                state=state,
                code="cancelled",
                message="Capability execution was cancelled.",
            )
            state.evidence.append(
                LifecycleEvidence(event=LifecycleEventKind.CANCELLATION, reason="Capability execution was cancelled.")
            )
            return await self._finish_async(
                case=case,
                conversation_id=conversation_id,
                state=state,
                outcome=CapabilityOutcome.CANCELLED,
                reason=CapabilityTerminationReason.CANCELLATION,
                detail="Capability execution was cancelled.",
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            state.evidence.append(ErrorEvidence(phase="executor", code=type(error).__name__, message=str(error)))
            return await self._finish_async(
                case=case,
                conversation_id=conversation_id,
                state=state,
                outcome=CapabilityOutcome.FAILED,
                reason=CapabilityTerminationReason.FAILURE,
                detail=str(error),
            )

    async def _within_wall_clock_async(
        self,
        *,
        operation: Awaitable[_AwaitedT],
        case: CapabilityCase,
        state: _ExecutionState,
    ) -> _AwaitedT:
        elapsed = (datetime.now(tz=timezone.utc) - state.started_at).total_seconds()
        remaining = max(0, case.task.limits.max_wall_clock_seconds - elapsed)
        try:
            return await asyncio.wait_for(operation, timeout=remaining)
        except asyncio.TimeoutError as error:
            raise _WallClockLimitError from error

    async def _persist_initial_messages_async(
        self,
        *,
        messages: tuple[Message, ...],
        conversation_id: str,
        state: _ExecutionState,
    ) -> Message:
        materialized = tuple(self._materialize_initial_message(message) for message in messages)
        for message in materialized[:-1]:
            persisted = await self._normalizer.persist_message_async(
                message=message,
                target=self._target,
                conversation_id=conversation_id,
            )
            state.message_piece_ids.extend(piece.id for piece in persisted.message_pieces)
        return materialized[-1]

    async def _send_generation_async(
        self,
        *,
        message: Message,
        conversation_id: str,
        declarations: tuple[ToolDeclaration, ...],
        request_already_persisted: bool,
    ) -> Message:
        options = self._request_options_factory.build_request_options(
            declarations=declarations,
            execution_policy=self._execution_policy,
        )
        return await self._normalizer.send_prompt_async(
            message=message,
            target=self._target,
            conversation_id=conversation_id,
            request_options=options,
            persist_request_before_send=not request_already_persisted,
            request_already_persisted=request_already_persisted,
        )

    async def _execute_calls_async(
        self,
        *,
        calls: tuple[PreparedToolCall, ...],
        case: CapabilityCase,
        conversation_id: str,
        cancellation_event: asyncio.Event | None,
        state: _ExecutionState,
    ) -> tuple[ToolExecutionRecord, ...]:
        async def execute_async(index: int, call: PreparedToolCall) -> ToolExecutionRecord:
            record = await self._tool_runtime.execute_call_async(
                call=call,
                case_id=case.case_id,
                conversation_id=conversation_id,
                asset_references=case.task.asset_references,
                environment_requirement_references=case.task.environment_requirement_references,
                cancellation_event=cancellation_event,
            )
            state.partial_tool_records[index] = record
            return record

        if self._execution_policy is ToolExecutionPolicy.SEQUENTIAL:
            return tuple([await execute_async(index, call) for index, call in enumerate(calls)])
        tasks = [asyncio.create_task(execute_async(index, call)) for index, call in enumerate(calls)]
        try:
            return tuple(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _persist_interrupted_tool_results_async(
        self,
        *,
        conversation_id: str,
        state: _ExecutionState,
        code: str,
        message: str,
    ) -> None:
        if not state.pending_calls:
            return
        records = []
        for index, call in enumerate(state.pending_calls):
            record = state.partial_tool_records.get(index)
            if record is None:
                error = ToolCallError(code=code, message=message)
                record = self._tool_runtime.not_executed_record(
                    call=call,
                    error=error,
                    phase="tool_execution",
                )
            records.append(record)
        result_message = self._result_message(records=tuple(records), state=state)
        await self._persist_result_message_async(
            message=result_message,
            conversation_id=conversation_id,
            state=state,
        )
        state.pending_calls = ()
        state.partial_tool_records.clear()

    def _deny_batch(self, *, calls: tuple[PreparedToolCall, ...]) -> tuple[ToolExecutionRecord, ...]:
        records = []
        for call in calls:
            if call.validation_error is not None:
                error = call.validation_error
            elif call.denied:
                error = ToolCallError(
                    code="approval_denied",
                    message=(
                        call.approval.reason
                        if call.approval and call.approval.reason
                        else "Tool call denied by policy."
                    ),
                )
            else:
                error = ToolCallError(
                    code="batch_denied",
                    message="Tool call was not executed because another call in the generation was denied.",
                )
            records.append(self._tool_runtime.not_executed_record(call=call, error=error, phase="tool_approval"))
        return tuple(records)

    def _result_message(
        self,
        *,
        records: tuple[ToolExecutionRecord, ...],
        state: _ExecutionState,
    ) -> Message:
        pieces = [
            MessagePiece(
                role="tool",
                original_value=record.result.to_json(),
                original_value_data_type="function_call_output",
                converted_value_data_type="function_call_output",
            )
            for record in records
        ]
        for record, piece in zip(records, pieces, strict=True):
            state.evidence.extend(
                evidence.model_copy(update={"result_piece_id": piece.id}) for evidence in record.execution_evidence
            )
            state.evidence.extend(record.error_evidence)
            state.evidence.extend(record.artifact_evidence)
            state.evidence.extend(record.additional_evidence)
        return Message(message_pieces=pieces)

    async def _persist_result_message_async(
        self,
        *,
        message: Message,
        conversation_id: str,
        state: _ExecutionState,
    ) -> None:
        persisted = await self._normalizer.persist_message_async(
            message=message,
            target=self._target,
            conversation_id=conversation_id,
        )
        state.message_piece_ids.extend(piece.id for piece in persisted.message_pieces)

    async def _finish_async(
        self,
        *,
        case: CapabilityCase,
        conversation_id: str,
        state: _ExecutionState,
        outcome: CapabilityOutcome,
        reason: CapabilityTerminationReason,
        detail: str | None,
    ) -> CapabilityTaskResult:
        result = CapabilityTaskResult(
            case_id=case.case_id,
            conversation_id=conversation_id,
            target_identifier=case.target_identifier,
            outcome=outcome,
            termination_reason=reason,
            termination_detail=detail,
            message_piece_ids=tuple(state.message_piece_ids),
            final_message_piece_ids=tuple(state.final_message_piece_ids),
            evidence=tuple(state.evidence),
            usage=state.usage,
            turns=state.model_generations,
            model_generations=state.model_generations,
            tool_calls=state.tool_calls,
            started_at=state.started_at,
            ended_at=datetime.now(tz=timezone.utc),
        )
        scores = []
        scoring_errors = []
        for scorer in self._scorers:
            try:
                scores.extend(await scorer.score_result_async(result=result, objective=case.task.objective))
            except Exception as error:
                scoring_errors.append(
                    ErrorEvidence(
                        phase="scoring",
                        code=type(error).__name__,
                        message=str(error),
                    )
                )
        return result.model_copy(
            update={
                "scores": tuple(scores),
                "evidence": (*result.evidence, *scoring_errors),
            }
        )

    def _record_generation_metadata(
        self,
        *,
        request_piece_id: uuid.UUID,
        response: Message,
        state: _ExecutionState,
    ) -> None:
        invocation = self._find_invocation(request_piece_id=request_piece_id)
        usages = (
            [metadata.usage for metadata in invocation.responses if metadata.usage is not None] if invocation else []
        )
        if not usages:
            fallback = TokenUsage.from_metadata(response.message_pieces[0].prompt_metadata)
            usages = [fallback] if fallback is not None else []
        for usage in usages:
            state.usage = state.usage.add(usage)
        if invocation and any(metadata.stop_reason == "length" for metadata in invocation.responses):
            state.evidence.append(
                LifecycleEvidence(event=LifecycleEventKind.TRUNCATION, reason="Target output token limit reached.")
            )

    def _find_invocation(self, *, request_piece_id: uuid.UUID) -> TargetInvocation | None:
        pieces = self._normalizer.memory.get_message_pieces(prompt_ids=[request_piece_id])
        if not pieces:
            return None
        return TargetInvocation.from_metadata(metadata=pieces[0].prompt_metadata)

    @staticmethod
    def _extract_tool_calls(
        *,
        response: Message,
    ) -> tuple[tuple[ToolCallRequest, uuid.UUID], ...]:
        return tuple(
            (ToolCallRequest.from_json(piece.converted_value), piece.id)
            for piece in response.message_pieces
            if piece.converted_value_data_type in ("function_call", "tool_call")
        )

    @staticmethod
    def _response_termination(
        *,
        response: Message,
    ) -> tuple[CapabilityOutcome, CapabilityTerminationReason, str | None] | None:
        if any(piece.is_truncated for piece in response.message_pieces):
            return CapabilityOutcome.INCOMPLETE, CapabilityTerminationReason.LIMIT, "Target output was truncated."
        if response.is_error():
            return CapabilityOutcome.FAILED, CapabilityTerminationReason.FAILURE, "Target generation returned an error."
        return None

    @staticmethod
    def _error_results_message(
        *,
        calls: tuple[tuple[ToolCallRequest, uuid.UUID], ...],
        code: str,
        message: str,
    ) -> Message:
        return Message(
            message_pieces=[
                MessagePiece(
                    role="tool",
                    original_value=ToolCallResult(
                        call_id=request.call_id,
                        output=message,
                        error=ToolCallError(code=code, message=message),
                    ).to_json(),
                    original_value_data_type="function_call_output",
                    converted_value_data_type="function_call_output",
                )
                for request, _ in calls
            ]
        )

    @staticmethod
    def _update_progress(*, records: tuple[ToolExecutionRecord, ...], state: _ExecutionState) -> None:
        had_success = any(record.result.error is None for record in records)
        had_error = any(record.result.error is not None for record in records)
        state.consecutive_no_progress = 0 if had_success else state.consecutive_no_progress + 1
        state.consecutive_errors = state.consecutive_errors + 1 if had_error and not had_success else 0

    @staticmethod
    def _tool_call_limit(
        *,
        case: CapabilityCase,
        state: _ExecutionState,
        additional_calls: int,
    ) -> str | None:
        limit = case.task.limits.max_tool_calls
        if state.tool_calls + additional_calls > limit:
            return f"Tool-call limit of {limit} would be exceeded."
        return None

    @staticmethod
    def _consecutive_limit(*, case: CapabilityCase, state: _ExecutionState) -> str | None:
        limits = case.task.limits
        if state.consecutive_no_progress >= limits.max_consecutive_no_progress:
            return f"Consecutive no-progress limit of {limits.max_consecutive_no_progress} reached."
        if state.consecutive_errors >= limits.max_consecutive_errors:
            return f"Consecutive error limit of {limits.max_consecutive_errors} reached."
        return None

    @staticmethod
    def _pre_generation_limit(
        *,
        case: CapabilityCase,
        state: _ExecutionState,
    ) -> tuple[CapabilityOutcome, CapabilityTerminationReason, str] | None:
        limits = case.task.limits
        if limits.max_turns is not None and state.model_generations >= limits.max_turns:
            detail = f"Turn limit of {limits.max_turns} reached."
        elif state.model_generations >= limits.max_model_generations:
            detail = f"Model-generation limit of {limits.max_model_generations} reached."
        elif (datetime.now(tz=timezone.utc) - state.started_at).total_seconds() >= limits.max_wall_clock_seconds:
            detail = f"Wall-clock limit of {limits.max_wall_clock_seconds} seconds reached."
        else:
            detail = CapabilityTaskExecutor._usage_limit(case=case, state=state)
            if detail is None:
                return None
        return CapabilityOutcome.INCOMPLETE, CapabilityTerminationReason.LIMIT, detail

    @staticmethod
    def _usage_limit(*, case: CapabilityCase, state: _ExecutionState) -> str | None:
        limits = case.task.limits
        if limits.max_input_tokens is not None and state.usage.input_tokens >= limits.max_input_tokens:
            return f"Input-token limit of {limits.max_input_tokens} reached."
        if limits.max_output_tokens is not None and state.usage.output_tokens >= limits.max_output_tokens:
            return f"Output-token limit of {limits.max_output_tokens} reached."
        if limits.max_total_tokens is not None and state.usage.total_tokens >= limits.max_total_tokens:
            return f"Total-token limit of {limits.max_total_tokens} reached."
        return None

    def _validate_case(self, *, case: CapabilityCase) -> None:
        actual = TargetIdentifier.from_component_identifier(self._target.get_identifier())
        if actual.hash != case.target_identifier.hash:
            raise ValueError("Capability case target does not match this executor's target.")

    @staticmethod
    def _raise_if_cancelled(*, cancellation_event: asyncio.Event | None) -> None:
        if cancellation_event is not None and cancellation_event.is_set():
            raise CooperativeCancellationError

    @staticmethod
    def _materialize_initial_message(message: Message) -> Message:
        materialized = message.duplicate()
        for piece in materialized.message_pieces:
            piece.original_prompt_id = piece.id
            piece.sequence = -1
            piece.conversation_id = None
        return materialized
