# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from pyrit.exceptions import RateLimitException
from pyrit.executor.capability import (
    ApprovalDecisionKind,
    ArtifactEvidence,
    CapabilityLimits,
    CapabilityOutcome,
    CapabilityTask,
    CapabilityTaskExecutor,
    CapabilityTerminationReason,
    CapabilityToolRuntime,
    ErrorEvidence,
    LifecycleEventKind,
    MessageScorerAdapter,
    OpenAIResponsesCapabilityRequestOptionsFactory,
    ToolApprovalDecision,
    ToolArtifact,
    ToolDeclaration,
    ToolExecutionContext,
    ToolExecutionError,
    ToolExecutionOutput,
    ToolExecutionPolicy,
    ToolExecutionStatus,
    ToolRegistry,
)
from pyrit.executor.capability.executor import _root_exception_name
from pyrit.models import (
    JSONValue,
    Message,
    MessagePiece,
    Score,
    TargetResponseMetadata,
    TokenUsage,
    ToolCallRequest,
    ToolCallResult,
)
from pyrit.prompt_target import (
    OpenAIResponsesRequestOptions,
    PromptTarget,
    TargetCapabilities,
    TargetConfiguration,
    TargetRequestOptions,
)
from pyrit.score import Scorer

if TYPE_CHECKING:
    from collections.abc import Mapping

pytestmark = pytest.mark.usefixtures("patch_central_database")


def test_root_exception_name_unwraps_only_generic_wrappers() -> None:
    semantic = RateLimitException()
    wrapper = Exception("normalizer wrapper")
    wrapper.__cause__ = semantic
    assert _root_exception_name(wrapper) == "RateLimitException"
    assert _root_exception_name(semantic) == "RateLimitException"


class FakeRequestOptionsFactory:
    def build_request_options(
        self,
        *,
        declarations: tuple[ToolDeclaration, ...],
        execution_policy: ToolExecutionPolicy,
    ) -> TargetRequestOptions:
        self.declarations = declarations
        self.execution_policy = execution_policy
        return TargetRequestOptions()


class FakeCapabilityTarget(PromptTarget):
    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_multi_message_pieces=True,
            supports_system_prompt=True,
            supports_editable_history=True,
            input_modalities=frozenset(
                {
                    frozenset({"text"}),
                    frozenset({"function_call_output"}),
                }
            ),
        )
    )

    def __init__(
        self,
        *,
        responses: list[Message | BaseException],
        usages: list[TokenUsage | None] | None = None,
        before_generation: Any | None = None,
    ) -> None:
        super().__init__()
        self._responses = list(responses)
        self._usages = list(usages or [])
        self._before_generation = before_generation
        self.normalized_conversations: list[list[Message]] = []

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        self.normalized_conversations.append(normalized_conversation)
        generation = len(self.normalized_conversations)
        if self._before_generation is not None:
            self._before_generation(generation)
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        conversation_id = normalized_conversation[-1].conversation_id
        for piece in item.message_pieces:
            piece.conversation_id = conversation_id
        usage = self._usages[generation - 1] if generation <= len(self._usages) else None
        stop_reason = "tool_calls" if _tool_requests(item) else "completed"
        self._record_response_metadata(
            metadata=TargetResponseMetadata(
                provider_response_id=f"generation-{generation}",
                stop_reason=stop_reason,
                provider_stop_reason=stop_reason,
                usage=usage,
            )
        )
        return [item]

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        return


@dataclass
class RecordingTool:
    outputs: list[ToolExecutionOutput] = field(default_factory=list)
    failures: list[ToolExecutionError] = field(default_factory=list)
    delay_seconds: float = 0
    calls: list[ToolExecutionContext] = field(default_factory=list)

    async def execute_async(
        self,
        *,
        arguments: Mapping[str, JSONValue],
        context: ToolExecutionContext,
    ) -> ToolExecutionOutput:
        self.calls.append(context)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.failures:
            raise self.failures.pop(0)
        if self.outputs:
            return self.outputs.pop(0)
        return ToolExecutionOutput(output={"arguments": dict(arguments)})


@dataclass
class ConcurrencyProbe:
    active: int = 0
    maximum: int = 0


@dataclass
class ConcurrentTool:
    probe: ConcurrencyProbe
    delay_seconds: float

    async def execute_async(
        self,
        *,
        arguments: Mapping[str, JSONValue],
        context: ToolExecutionContext,
    ) -> ToolExecutionOutput:
        self.probe.active += 1
        self.probe.maximum = max(self.probe.maximum, self.probe.active)
        await asyncio.sleep(self.delay_seconds)
        self.probe.active -= 1
        return ToolExecutionOutput(output={"call_id": context.call_id})


class DenyNamedToolPolicy:
    async def decide_async(
        self,
        *,
        request: ToolCallRequest,
        declaration: ToolDeclaration,
        arguments: Mapping[str, JSONValue],
        context: ToolExecutionContext,
    ) -> ToolApprovalDecision:
        decision = ApprovalDecisionKind.DENY if request.name == "blocked" else ApprovalDecisionKind.ALLOW
        return ToolApprovalDecision(
            decision=decision,
            policy=type(self).__name__,
            reason="policy" if decision is ApprovalDecisionKind.DENY else None,
        )


def _text_message(text: str = "done") -> Message:
    return Message.from_prompt(prompt=text, role="assistant")


def _tool_call_message(*calls: tuple[str, str, str]) -> Message:
    return Message(
        message_pieces=[
            MessagePiece(
                role="assistant",
                original_value=ToolCallRequest(call_id=call_id, name=name, arguments=arguments).to_json(),
                original_value_data_type="function_call",
                converted_value_data_type="function_call",
            )
            for call_id, name, arguments in calls
        ]
    )


def _tool_requests(message: Message) -> list[ToolCallRequest]:
    return [
        ToolCallRequest.from_json(piece.converted_value)
        for piece in message.message_pieces
        if piece.converted_value_data_type in ("function_call", "tool_call")
    ]


def _tool_results(target: FakeCapabilityTarget, *, conversation_id: str) -> list[tuple[MessagePiece, ToolCallResult]]:
    pieces = target._memory.get_message_pieces(conversation_id=conversation_id)
    return [
        (piece, ToolCallResult.from_json(piece.converted_value))
        for piece in pieces
        if piece.converted_value_data_type == "function_call_output"
    ]


def _registry(*bindings: tuple[ToolDeclaration, RecordingTool | ConcurrentTool]) -> ToolRegistry:
    registry = ToolRegistry()
    for declaration, implementation in bindings:
        registry.register(declaration=declaration, implementation=implementation)
    return registry


def _executor(
    *,
    target: FakeCapabilityTarget,
    registry: ToolRegistry | None = None,
    runtime: CapabilityToolRuntime | None = None,
    policy: ToolExecutionPolicy = ToolExecutionPolicy.SEQUENTIAL,
    scorers: tuple[Any, ...] = (),
) -> CapabilityTaskExecutor:
    registry = registry or ToolRegistry()
    return CapabilityTaskExecutor(
        target=target,
        tool_registry=registry,
        request_options_factory=FakeRequestOptionsFactory(),
        tool_runtime=runtime,
        execution_policy=policy,
        scorers=scorers,
    )


def test_openai_options_adapter_forces_single_generation() -> None:
    options = OpenAIResponsesCapabilityRequestOptionsFactory().build_request_options(
        declarations=(
            ToolDeclaration(
                name="lookup",
                description="Look up a value.",
                input_schema={"type": "object", "properties": {"key": {"type": "string"}}},
            ),
        ),
        execution_policy=ToolExecutionPolicy.PARALLEL,
    )

    assert isinstance(options, OpenAIResponsesRequestOptions)
    assert options.tool_execution_mode == "single_generation"
    assert options.parallel_tool_calls is True
    assert options.tools is not None
    assert options.tools[0].name == "lookup"


async def test_simple_completion_and_usage_accounting() -> None:
    target = FakeCapabilityTarget(
        responses=[_text_message()],
        usages=[TokenUsage(input_tokens=4, output_tokens=3, total_tokens=7)],
    )

    result = await _executor(target=target).execute_task_async(task=CapabilityTask(objective="finish"))

    assert result.outcome is CapabilityOutcome.COMPLETED
    assert result.termination_reason is CapabilityTerminationReason.COMPLETION
    assert result.model_generations == 1
    assert result.turns == 1
    assert result.usage.total_tokens == 7
    assert result.final_message_piece_ids


async def test_task_with_initial_messages_can_be_reused() -> None:
    target = FakeCapabilityTarget(responses=[_text_message("first"), _text_message("second")])
    executor = _executor(target=target)
    task = CapabilityTask(
        objective="reuse",
        initial_messages=(Message.from_prompt(prompt="same template", role="user"),),
    )

    first = await executor.execute_task_async(task=task)
    second = await executor.execute_task_async(task=task)

    assert first.outcome is CapabilityOutcome.COMPLETED
    assert second.outcome is CapabilityOutcome.COMPLETED
    assert set(first.message_piece_ids).isdisjoint(second.message_piece_ids)


async def test_multi_generation_usage_counts_each_invocation_once() -> None:
    tool = RecordingTool()
    registry = _registry((ToolDeclaration(name="lookup"), tool))
    target = FakeCapabilityTarget(
        responses=[_tool_call_message(("call-1", "lookup", "{}")), _text_message()],
        usages=[
            TokenUsage(input_tokens=4, output_tokens=6, total_tokens=10),
            TokenUsage(input_tokens=12, output_tokens=8, total_tokens=20),
        ],
    )

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(objective="account")
    )

    assert result.usage.input_tokens == 16
    assert result.usage.output_tokens == 14
    assert result.usage.total_tokens == 30


async def test_sequential_multi_call_preserves_result_order() -> None:
    first = RecordingTool(outputs=[ToolExecutionOutput(output="first")])
    second = RecordingTool(outputs=[ToolExecutionOutput(output="second")])
    registry = _registry(
        (ToolDeclaration(name="first"), first),
        (ToolDeclaration(name="second"), second),
    )
    target = FakeCapabilityTarget(
        responses=[
            _tool_call_message(("call-1", "first", "{}"), ("call-2", "second", "{}")),
            _text_message(),
        ]
    )

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(objective="use tools")
    )

    persisted = _tool_results(target, conversation_id=result.conversation_id)
    assert [tool_result.call_id for _, tool_result in persisted] == ["call-1", "call-2"]
    assert [tool_result.output for _, tool_result in persisted] == ["first", "second"]
    assert len(first.calls) == len(second.calls) == 1


async def test_parallel_multi_call_executes_concurrently_and_preserves_order() -> None:
    probe = ConcurrencyProbe()
    registry = _registry(
        (ToolDeclaration(name="slow"), ConcurrentTool(probe=probe, delay_seconds=0.02)),
        (ToolDeclaration(name="fast"), ConcurrentTool(probe=probe, delay_seconds=0.001)),
    )
    target = FakeCapabilityTarget(
        responses=[
            _tool_call_message(("slow-call", "slow", "{}"), ("fast-call", "fast", "{}")),
            _text_message(),
        ]
    )

    result = await _executor(
        target=target,
        registry=registry,
        policy=ToolExecutionPolicy.PARALLEL,
    ).execute_task_async(task=CapabilityTask(objective="parallel"))

    persisted = _tool_results(target, conversation_id=result.conversation_id)
    assert probe.maximum == 2
    assert [tool_result.call_id for _, tool_result in persisted] == ["slow-call", "fast-call"]


async def test_denial_prevents_entire_batch_execution() -> None:
    allowed = RecordingTool()
    blocked = RecordingTool()
    registry = _registry(
        (ToolDeclaration(name="allowed"), allowed),
        (ToolDeclaration(name="blocked"), blocked),
    )
    runtime = CapabilityToolRuntime(registry=registry, approval_policy=DenyNamedToolPolicy())
    target = FakeCapabilityTarget(
        responses=[_tool_call_message(("call-1", "allowed", "{}"), ("call-2", "blocked", "{}"))]
    )

    result = await _executor(target=target, registry=registry, runtime=runtime).execute_task_async(
        task=CapabilityTask(objective="denial")
    )

    assert result.outcome is CapabilityOutcome.DENIED
    assert result.termination_reason is CapabilityTerminationReason.DENIAL
    assert not allowed.calls
    assert not blocked.calls
    persisted = _tool_results(target, conversation_id=result.conversation_id)
    assert [tool_result.error.code for _, tool_result in persisted if tool_result.error] == [
        "batch_denied",
        "approval_denied",
    ]


@pytest.mark.parametrize(
    ("call", "expected_code"),
    [
        (("missing-call", "missing", "{}"), "missing_tool"),
        (("malformed-call", "declared", "{"), "malformed_arguments"),
    ],
)
async def test_missing_and_malformed_tools_return_deterministic_results(
    call: tuple[str, str, str],
    expected_code: str,
) -> None:
    registry = ToolRegistry()
    registry.register(
        declaration=ToolDeclaration(name="declared"),
        implementation=RecordingTool(),
    )
    target = FakeCapabilityTarget(responses=[_tool_call_message(call), _text_message()])

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(objective="invalid call")
    )

    persisted = _tool_results(target, conversation_id=result.conversation_id)
    assert persisted[0][1].error is not None
    assert persisted[0][1].error.code == expected_code
    assert any(
        evidence.status is ToolExecutionStatus.NOT_EXECUTED and evidence.error_code == expected_code
        for evidence in result.evidence
        if evidence.evidence_type == "tool_execution"
    )


async def test_tool_timeout_returns_typed_error_and_evidence() -> None:
    tool = RecordingTool(delay_seconds=0.05)
    declaration = ToolDeclaration(name="slow", timeout_seconds=0.001)
    registry = _registry((declaration, tool))
    target = FakeCapabilityTarget(responses=[_tool_call_message(("call-1", "slow", "{}")), _text_message()])

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(objective="timeout")
    )

    persisted = _tool_results(target, conversation_id=result.conversation_id)
    assert persisted[0][1].error is not None
    assert persisted[0][1].error.code == "timeout"
    execution = [e for e in result.evidence if e.evidence_type == "tool_execution"]
    assert execution[0].status is ToolExecutionStatus.TIMED_OUT


async def test_cancellation_event_cancels_running_tool() -> None:
    tool = RecordingTool(delay_seconds=1)
    registry = _registry((ToolDeclaration(name="slow"), tool))
    target = FakeCapabilityTarget(responses=[_tool_call_message(("call-1", "slow", "{}"))])
    cancellation_event = asyncio.Event()

    execution = asyncio.create_task(
        _executor(target=target, registry=registry).execute_task_async(
            task=CapabilityTask(objective="cancel"),
            cancellation_event=cancellation_event,
        )
    )
    while not tool.calls:
        await asyncio.sleep(0)
    cancellation_event.set()
    result = await execution

    assert result.outcome is CapabilityOutcome.CANCELLED
    assert result.termination_reason is CapabilityTerminationReason.CANCELLATION
    assert any(
        evidence.event is LifecycleEventKind.CANCELLATION
        for evidence in result.evidence
        if evidence.evidence_type == "lifecycle"
    )


async def test_cancellation_after_completed_tool_keeps_persisted_result() -> None:
    class CancellingTool:
        async def execute_async(
            self,
            *,
            arguments: Mapping[str, JSONValue],
            context: ToolExecutionContext,
        ) -> ToolExecutionOutput:
            assert context.cancellation_event is not None
            context.cancellation_event.set()
            return ToolExecutionOutput(output="completed", side_effect_completed=True)

    registry = ToolRegistry()
    registry.register(declaration=ToolDeclaration(name="cancel"), implementation=CancellingTool())
    target = FakeCapabilityTarget(responses=[_tool_call_message(("call-1", "cancel", "{}"))])
    cancellation_event = asyncio.Event()

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(objective="cancel after"),
        cancellation_event=cancellation_event,
    )

    persisted = _tool_results(target, conversation_id=result.conversation_id)
    assert result.outcome is CapabilityOutcome.CANCELLED
    assert persisted[0][1].output == "completed"
    execution = [e for e in result.evidence if e.evidence_type == "tool_execution"]
    assert execution[0].result_piece_id in result.message_piece_ids


async def test_external_task_cancellation_propagates() -> None:
    tool = RecordingTool(delay_seconds=1)
    registry = _registry((ToolDeclaration(name="slow"), tool))
    target = FakeCapabilityTarget(responses=[_tool_call_message(("call-1", "slow", "{}"))])
    execution = asyncio.create_task(
        _executor(target=target, registry=registry).execute_task_async(
            task=CapabilityTask(objective="cancel externally")
        )
    )
    while not tool.calls:
        await asyncio.sleep(0)

    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution
    assert execution.cancelled()


async def test_unexpected_parallel_tool_error_produces_results_for_entire_batch() -> None:
    class UnexpectedTool:
        async def execute_async(
            self,
            *,
            arguments: Mapping[str, JSONValue],
            context: ToolExecutionContext,
        ) -> ToolExecutionOutput:
            raise ValueError("unexpected")

    sibling = RecordingTool(
        delay_seconds=0.01,
        outputs=[ToolExecutionOutput(output="completed", side_effect_completed=True)],
    )
    registry = ToolRegistry()
    registry.register(declaration=ToolDeclaration(name="unexpected"), implementation=UnexpectedTool())
    registry.register(declaration=ToolDeclaration(name="sibling"), implementation=sibling)
    target = FakeCapabilityTarget(
        responses=[
            _tool_call_message(("call-1", "unexpected", "{}"), ("call-2", "sibling", "{}")),
            _text_message(),
        ]
    )

    result = await _executor(
        target=target,
        registry=registry,
        policy=ToolExecutionPolicy.PARALLEL,
    ).execute_task_async(task=CapabilityTask(objective="contain errors"))

    persisted = _tool_results(target, conversation_id=result.conversation_id)
    assert result.outcome is CapabilityOutcome.COMPLETED
    assert len(persisted) == 2
    assert persisted[0][1].error is not None
    assert persisted[0][1].error.code == "implementation_error"
    assert persisted[1][1].output == "completed"
    assert len(sibling.calls) == 1


async def test_truncated_tool_generation_never_executes_call() -> None:
    tool = RecordingTool()
    registry = _registry((ToolDeclaration(name="unsafe"), tool))
    call_message = _tool_call_message(("call-1", "unsafe", "{}"))
    call_message.get_piece().mark_as_truncated()
    target = FakeCapabilityTarget(responses=[call_message])

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(objective="truncate")
    )

    assert result.termination_reason is CapabilityTerminationReason.LIMIT
    assert not tool.calls
    persisted = _tool_results(target, conversation_id=result.conversation_id)
    assert persisted[0][1].error is not None
    assert persisted[0][1].error.code == "generation_incomplete"


async def test_truncated_completion_agent_response_does_not_continue() -> None:
    response = _text_message("partial")
    response.get_piece().mark_as_truncated()
    target = FakeCapabilityTarget(responses=[response])

    result = await _executor(target=target).execute_task_async(
        task=CapabilityTask(objective="truncate", completion_tool_name="submit")
    )

    assert result.termination_reason is CapabilityTerminationReason.LIMIT
    assert result.model_generations == 1


async def test_idempotent_tool_retries_declared_retryable_failure() -> None:
    tool = RecordingTool(
        failures=[
            ToolExecutionError(code="transient", message="retry", retryable=True),
        ],
        outputs=[ToolExecutionOutput(output="ok")],
    )
    declaration = ToolDeclaration(
        name="retryable",
        idempotent=True,
        max_retries=2,
        retryable_error_codes=("transient",),
    )
    registry = _registry((declaration, tool))
    target = FakeCapabilityTarget(responses=[_tool_call_message(("stable-call", "retryable", "{}")), _text_message()])

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(objective="retry")
    )

    attempts = [e for e in result.evidence if e.evidence_type == "tool_execution"]
    assert len(tool.calls) == 2
    assert [attempt.attempt_number for attempt in attempts] == [1, 2]
    assert {attempt.call_id for attempt in attempts} == {"stable-call"}
    assert len({attempt.attempt_id for attempt in attempts}) == 2


@pytest.mark.parametrize("side_effect_completed", [False, True])
async def test_non_replayable_tool_failure_runs_once(side_effect_completed: bool) -> None:
    tool = RecordingTool(
        failures=[
            ToolExecutionError(
                code="transient",
                message="do not replay",
                retryable=True,
                side_effect_completed=side_effect_completed,
            )
        ]
    )
    declaration = ToolDeclaration(
        name="unsafe",
        idempotent=side_effect_completed,
        max_retries=3,
        retryable_error_codes=("transient",),
    )
    registry = _registry((declaration, tool))
    target = FakeCapabilityTarget(responses=[_tool_call_message(("call-1", "unsafe", "{}")), _text_message()])

    await _executor(target=target, registry=registry).execute_task_async(task=CapabilityTask(objective="no replay"))

    assert len(tool.calls) == 1


async def test_tool_result_is_persisted_before_next_generation() -> None:
    tool = RecordingTool()
    registry = _registry((ToolDeclaration(name="lookup"), tool))
    target = FakeCapabilityTarget(responses=[_tool_call_message(("call-1", "lookup", "{}")), _text_message()])

    def assert_persisted(generation: int) -> None:
        if generation != 2:
            return
        conversation_id = target.normalized_conversations[0][-1].conversation_id
        assert any(
            piece.converted_value_data_type == "function_call_output"
            for piece in target._memory.get_message_pieces(conversation_id=conversation_id)
        )

    target._before_generation = assert_persisted
    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(objective="persist first")
    )

    assert result.outcome is CapabilityOutcome.COMPLETED


async def test_later_generation_failure_does_not_replay_side_effect() -> None:
    tool = RecordingTool(outputs=[ToolExecutionOutput(output="done", side_effect_completed=True)])
    registry = _registry((ToolDeclaration(name="side_effect"), tool))
    target = FakeCapabilityTarget(
        responses=[
            _tool_call_message(("call-1", "side_effect", "{}")),
            RuntimeError("provider failed"),
        ]
    )

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(objective="fail later")
    )

    assert result.outcome is CapabilityOutcome.FAILED
    assert result.termination_reason is CapabilityTerminationReason.FAILURE
    assert len(tool.calls) == 1
    assert len(_tool_results(target, conversation_id=result.conversation_id)) == 1


@pytest.mark.parametrize(
    ("limits", "expected_detail"),
    [
        (CapabilityLimits(max_model_generations=1), "Model-generation limit"),
        (CapabilityLimits(max_turns=1), "Turn limit"),
        (CapabilityLimits(max_tool_calls=0), "Tool-call limit"),
    ],
)
async def test_count_limits_terminate_explicitly(
    limits: CapabilityLimits,
    expected_detail: str,
) -> None:
    tool = RecordingTool()
    registry = _registry((ToolDeclaration(name="tool"), tool))
    target = FakeCapabilityTarget(responses=[_tool_call_message(("call-1", "tool", "{}"))])

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(objective="limit", limits=limits)
    )

    assert result.termination_reason is CapabilityTerminationReason.LIMIT
    assert expected_detail in (result.termination_detail or "")
    persisted = _tool_results(target, conversation_id=result.conversation_id)
    assert len(persisted) == 1
    if limits.max_tool_calls == 0:
        assert not tool.calls
        assert result.tool_calls == 0
        assert persisted[0][1].error is not None
    else:
        execution = [e for e in result.evidence if e.evidence_type == "tool_execution"]
        assert execution[0].result_piece_id in result.message_piece_ids


async def test_wall_clock_limit_stops_after_slow_tool() -> None:
    tool = RecordingTool(delay_seconds=0.02)
    registry = _registry((ToolDeclaration(name="slow"), tool))
    target = FakeCapabilityTarget(responses=[_tool_call_message(("call-1", "slow", "{}"))])

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(
            objective="wall",
            limits=CapabilityLimits(max_wall_clock_seconds=0.001),
        )
    )

    assert result.termination_reason is CapabilityTerminationReason.LIMIT
    assert "Wall-clock limit" in (result.termination_detail or "")


async def test_wall_clock_limit_preserves_completed_tool_evidence() -> None:
    first = RecordingTool(outputs=[ToolExecutionOutput(output="first", side_effect_completed=True)])
    second = RecordingTool(delay_seconds=1)
    registry = _registry(
        (ToolDeclaration(name="first"), first),
        (ToolDeclaration(name="second"), second),
    )
    target = FakeCapabilityTarget(responses=[_tool_call_message(("call-1", "first", "{}"), ("call-2", "second", "{}"))])

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(
            objective="partial wall",
            limits=CapabilityLimits(max_wall_clock_seconds=0.2),
        )
    )

    persisted = _tool_results(target, conversation_id=result.conversation_id)
    executions = [e for e in result.evidence if e.evidence_type == "tool_execution"]
    assert result.termination_reason is CapabilityTerminationReason.LIMIT
    assert [tool_result.call_id for _, tool_result in persisted] == ["call-1", "call-2"]
    assert persisted[0][1].output == "first"
    assert persisted[1][1].error is not None
    assert persisted[1][1].error.code == "wall_clock_limit"
    assert executions[0].side_effect_completed is True
    assert all(evidence.result_piece_id in result.message_piece_ids for evidence in executions)


@pytest.mark.parametrize(
    ("limits", "expected_detail"),
    [
        (
            CapabilityLimits(max_consecutive_no_progress=1, max_consecutive_errors=10),
            "no-progress",
        ),
        (
            CapabilityLimits(max_consecutive_no_progress=10, max_consecutive_errors=1),
            "error limit",
        ),
    ],
)
async def test_consecutive_behavior_limits(
    limits: CapabilityLimits,
    expected_detail: str,
) -> None:
    registry = ToolRegistry()
    target = FakeCapabilityTarget(responses=[_tool_call_message(("call-1", "missing", "{}"))])

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(objective="stuck", limits=limits)
    )

    assert result.termination_reason is CapabilityTerminationReason.LIMIT
    assert expected_detail in (result.termination_detail or "")


async def test_usage_limit_applies_to_current_generation() -> None:
    target = FakeCapabilityTarget(
        responses=[_text_message()],
        usages=[TokenUsage(input_tokens=6, output_tokens=5, total_tokens=11)],
    )

    result = await _executor(target=target).execute_task_async(
        task=CapabilityTask(
            objective="tokens",
            limits=CapabilityLimits(max_total_tokens=10),
        )
    )

    assert result.termination_reason is CapabilityTerminationReason.LIMIT
    assert result.usage.total_tokens == 11


async def test_usage_limit_closes_out_requested_tool_calls() -> None:
    tool = RecordingTool()
    registry = _registry((ToolDeclaration(name="unused"), tool))
    target = FakeCapabilityTarget(
        responses=[_tool_call_message(("call-1", "unused", "{}"))],
        usages=[TokenUsage(total_tokens=11)],
    )

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(
            objective="tokens",
            limits=CapabilityLimits(max_total_tokens=10),
        )
    )

    persisted = _tool_results(target, conversation_id=result.conversation_id)
    assert result.tool_calls == 0
    assert not tool.calls
    assert persisted[0][1].error is not None
    assert persisted[0][1].error.code == "usage_limit"


async def test_evidence_links_request_result_attempt_and_artifact() -> None:
    artifact = ToolArtifact(artifact_id="artifact-1", uri="memory://artifact", sha256="abc")
    tool = RecordingTool(
        outputs=[
            ToolExecutionOutput(
                output="created",
                artifacts=(artifact,),
                side_effect_completed=True,
            )
        ]
    )
    registry = _registry((ToolDeclaration(name="create"), tool))
    target = FakeCapabilityTarget(responses=[_tool_call_message(("call-1", "create", "{}")), _text_message()])

    result = await _executor(target=target, registry=registry).execute_task_async(
        task=CapabilityTask(objective="evidence")
    )

    executions = [e for e in result.evidence if e.evidence_type == "tool_execution"]
    artifacts = [e for e in result.evidence if isinstance(e, ArtifactEvidence)]
    assert executions[0].request_piece_id in result.message_piece_ids
    assert executions[0].result_piece_id in result.message_piece_ids
    assert executions[0].call_id == "call-1"
    assert executions[0].side_effect_completed is True
    assert artifacts[0].created_by_call_id == "call-1"


async def test_existing_message_scorer_adapter_scores_final_message() -> None:
    target = FakeCapabilityTarget(responses=[_text_message("answer")])
    scorer = MagicMock(spec=Scorer)

    async def score_async(message: Message, *, objective: str) -> list[Score]:
        return [
            Score(
                score_value="true",
                score_type="true_false",
                message_piece_id=message.get_piece().id,
                objective=objective,
            )
        ]

    scorer.score_async = AsyncMock(side_effect=score_async)
    adapter = MessageScorerAdapter(scorer=scorer, memory=target._memory)

    result = await _executor(target=target, scorers=(adapter,)).execute_task_async(
        task=CapabilityTask(objective="score")
    )

    assert len(result.scores) == 1
    assert result.scores[0].message_piece_id == result.final_message_piece_ids[0]
    assert result.scores[0].objective == "score"


async def test_scorer_failure_is_evidence_not_result_loss() -> None:
    scorer = MagicMock()
    scorer.score_result_async = AsyncMock(side_effect=RuntimeError("scorer failed"))
    target = FakeCapabilityTarget(responses=[_text_message()])

    result = await _executor(target=target, scorers=(scorer,)).execute_task_async(
        task=CapabilityTask(objective="score safely")
    )

    assert result.outcome is CapabilityOutcome.COMPLETED
    assert not result.scores
    errors = [e for e in result.evidence if e.evidence_type == "error"]
    assert errors[-1].phase == "scoring"
    assert errors[-1].message == "scorer failed"


def test_evidence_discriminator_and_task_are_immutable() -> None:
    with pytest.raises(ValidationError):
        ErrorEvidence(evidence_type="artifact", phase="test", code="bad", message="bad")  # type: ignore[arg-type]

    task = CapabilityTask(objective="immutable")
    with pytest.raises(ValidationError):
        task.objective = "changed"
