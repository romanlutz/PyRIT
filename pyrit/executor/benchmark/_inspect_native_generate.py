# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pyrit.exceptions import get_retry_max_num_attempts
from pyrit.memory import CentralMemory, set_message_piece_sha256_async
from pyrit.models import Conversation, Message, MessagePiece, TokenUsage
from pyrit.prompt_normalizer import PromptNormalizer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from inspect_ai import Task
    from inspect_ai.model import ChatMessage, ChatMessageAssistant
    from inspect_ai.solver import TaskState
    from inspect_ai.tool import ToolDef

    from pyrit.executor.benchmark._inspect_artifacts import InspectRunArtifacts
    from pyrit.prompt_target import OpenAIResponseTarget

logger = logging.getLogger(__name__)


class InspectNativeLimitError(RuntimeError):
    """A local generation or tool bound stopped acquisition, not a behavior failure."""


class _NativeBudgetStopError(RuntimeError):
    def __init__(self, reason: str = "budget") -> None:
        super().__init__(reason)
        self.reason = reason


class InspectNativeGenerate:
    """Drive a native solver's Generate seam without taking over its tools or prompts."""

    def __init__(
        self,
        *,
        target_factory: Callable[[list[dict[str, Any]]], OpenAIResponseTarget],
        model_name: str,
        artifacts: InspectRunArtifacts,
        after_tool_async: Callable[[], Awaitable[bool]],
        on_tool_return: Callable[[], None] | None = None,
        max_requests: int,
        max_tool_calls: int,
        max_tool_output_bytes: int,
        max_response_bytes: int | None = None,
    ) -> None:
        """Bind a sequential external tool loop and explicit local limits."""
        self.target_factory = target_factory
        self.model_name = model_name
        self.artifacts = artifacts
        self.after_tool_async = after_tool_async
        self.on_tool_return = on_tool_return
        self.max_requests = max_requests
        self.max_tool_calls = max_tool_calls
        self.max_tool_output_bytes = max_tool_output_bytes
        self.max_response_bytes = max_response_bytes if max_response_bytes is not None else max_tool_output_bytes
        self.conversation_id = str(uuid4())
        self.requests = 0
        self.tool_calls: list[dict[str, Any]] = []
        self.usage: dict[str, int] = {}
        self.termination_reason: str | None = None
        self.target: OpenAIResponseTarget | None = None
        self._normalizer = PromptNormalizer()
        self._prefix: list[str] = []
        self._tool_schemas: list[dict[str, Any]] | None = None
        self._tool_functions: list[Callable[..., Any]] = []
        self._call_ids: set[str] = set()
        self._raw_arguments: dict[str, str] = {}
        self._persistence: dict[str, asyncio.Task[None]] = {}
        self._terminal = False
        self._callback_cancellation: asyncio.CancelledError | None = None
        self._native_turn_limit: int | None = None
        self._tool_dispatches = 0
        self._last_usage: TokenUsage | None = None
        self._missing_usage: set[str] = set()

    @property
    def callback_cancellation(self) -> asyncio.CancelledError | None:
        """The actual callback cancellation, retained before native tool error conversion."""
        return self._callback_cancellation

    def configure_native_limits(self, task: Task) -> None:
        """
        Bind enforceable native task limits before any callback.

        Raises:
            ValueError: If a native limit is invalid.
            NotImplementedError: If the external target cannot meter a configured limit.
        """
        if task.cost_limit is not None or task.working_limit is not None:
            raise NotImplementedError("External target cost and working-time metering are not supported.")
        if task.token_limit is not None and task.token_limit_type not in (None, "all", "output"):
            raise NotImplementedError("External token limits support only actual all/output token counts.")
        for limit in (task.turn_limit, task.message_limit, task.token_limit):
            if limit is not None and (type(limit) is not int or limit < 0):
                raise ValueError("Native generation, message, and token limits must be nonnegative integers.")
        self._native_turn_limit = task.turn_limit

    async def generate_async(
        self, state: TaskState, tool_calls: Literal["loop", "single", "none"] = "loop", **kwargs: Any
    ) -> TaskState:
        """
        Run the supported native loop with actual assistant and tool continuation messages.

        Returns:
            TaskState: The original state with actual provider/tool messages appended.

        Raises:
            ValueError: If the native task requests an unsupported generation mode or mutates history.
            RuntimeError: If the loop would continue after completion or exceed its explicit limits.
        """
        if tool_calls != "loop" or kwargs or state.tool_choice not in (None, "auto"):
            raise ValueError(
                "This strict bridge supports Generate(tool_calls='loop'), fixed tools, and auto choice only."
            )
        try:
            self._check_dispatch(state=state, provider=True)
            self._bind_tools(state)
            while True:
                self._check_dispatch(state=state, provider=True)
                response = await self._send_one_async(state)
                assistant = self._assistant_message(response)
                state.messages.append(assistant)
                self._prefix.append(self._message_key(assistant))
                self._set_output(state=state, assistant=assistant)
                if not assistant.tool_calls:
                    self._check_dispatch(state=state, provider=True)
                    return state
                self._check_dispatch(state=state, provider=False)
                await self._execute_tool_async(state)
                if await self.after_tool_async():
                    await self._retain_pending_async(state)
                    self._check_stopped()
                    state.completed = True
                    self._terminal = True
                    self.termination_reason = "full_success"
                    return state
        except _NativeBudgetStopError as budget:
            return await self._finish_budget_async(state, reason=budget.reason)

    async def retain_pending_async(self, state: TaskState) -> None:
        """Persist authentic task-added messages on termination without making a provider call."""
        await self._retain_pending_async(state)

    def stop(self) -> None:
        """Prevent future dispatch when the owning local evaluation is cancelled."""
        self._terminal = True

    async def _finish_budget_async(self, state: TaskState, *, reason: str = "budget") -> TaskState:
        await self._retain_pending_async(state)
        self._check_stopped()
        state.completed = True
        self._terminal = True
        self.termination_reason = reason
        return state

    def _check_stopped(self) -> None:
        if self._terminal:
            raise InspectNativeLimitError("The native Generate loop was stopped; no further dispatch is allowed.")

    def _check_dispatch(self, *, state: TaskState, provider: bool, reserved_tool: bool = False) -> None:
        self._check_stopped()
        self._check_memory()
        if state.cost_limit is not None or (
            state.token_limit is not None and state.token_limit_type not in ("all", "output")
        ):
            raise NotImplementedError(
                "External target supports only native all/output token metering, not cost/formulas."
            )
        if state.token_limit is not None and self.requests:
            required = "total_tokens" if state.token_limit_type == "all" else "output_tokens"
            if required in self._missing_usage:
                raise InspectNativeLimitError(
                    "Provider usage required for the configured native token limit is missing."
                )
        if state.message_limit is not None and len(state.messages) >= state.message_limit:
            raise _NativeBudgetStopError
        if state.completed:
            raise _NativeBudgetStopError("binding_state")
        if not reserved_tool and self._tool_dispatches >= self.max_tool_calls:
            raise _NativeBudgetStopError
        if provider:
            if self.requests >= self.max_requests or (
                self._native_turn_limit is not None and self.requests >= self._native_turn_limit
            ):
                raise _NativeBudgetStopError
            if state.token_limit is not None:
                field = "total_tokens" if state.token_limit_type == "all" else "output_tokens"
                if self.usage.get(field, 0) >= state.token_limit:
                    raise _NativeBudgetStopError

    def _bind_tools(self, state: TaskState) -> None:
        from inspect_ai.tool import ToolDef

        self._check_memory()
        definitions = [ToolDef(tool) for tool in state.tools]
        if any(definition.model_input is not None for definition in definitions):
            raise ValueError("Custom native tool model_input rendering is not supported by this string bridge.")
        schemas = [
            {
                "type": "function",
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.parameters.model_dump(exclude_none=True),
            }
            for definition in definitions
        ]
        names = [definition.name for definition in definitions]
        if len(names) != len(set(names)):
            raise ValueError("Native tool names must be unique.")
        functions = [definition.tool for definition in definitions]
        if self.target is not None:
            if schemas != self._tool_schemas or functions != self._tool_functions:
                raise ValueError("Changing native tools after target binding is unsupported.")
            return
        target = self.target_factory(schemas)
        if target.auto_execute_tools:
            raise ValueError("The strict bridge requires OpenAIResponseTarget(auto_execute_tools=False).")
        if get_retry_max_num_attempts() != 1:
            raise ValueError("Strict offline acquisition requires RETRY_MAX_NUM_ATTEMPTS=1.")
        configuration = target.get_identifier().params.get("extra_body_parameters")
        if (
            not isinstance(configuration, dict)
            or configuration.get("tools") != schemas
            or configuration.get("parallel_tool_calls") is not False
            or configuration.get("store") is not False
        ):
            raise ValueError(
                "The target must preserve native tool schemas, disable parallel calls, and set store=false."
            )
        self.target = target
        self._tool_schemas = schemas
        self._tool_functions = functions

    async def _send_one_async(self, state: TaskState) -> Message:
        self._check_dispatch(state=state, provider=True)
        if self.target is None:
            raise RuntimeError("Native tools must be bound before sending.")
        new_messages = self._new_messages(state)
        memory = CentralMemory.get_memory_instance()
        await asyncio.to_thread(
            memory.add_conversation_to_memory,
            conversation=Conversation(
                conversation_id=self.conversation_id, target_identifier=self.target.get_identifier()
            ),
        )
        for native in new_messages[:-1]:
            await self._persist_native_async(native)
        request = await self._native_message_async(new_messages[-1]) if new_messages else None
        await self.artifacts.append_async(
            event="native_generation_request",
            data={
                "ordinal": self.requests + 1,
                "conversation_id": self.conversation_id,
                "continuation": request is None,
                "native_messages": [message.model_dump(mode="json") for message in state.messages],
            },
        )
        self._check_dispatch(state=state, provider=True)
        self.requests += 1
        if request is None:
            response = await self._normalizer.continue_conversation_async(
                target=self.target, conversation_id=self.conversation_id
            )
        else:
            response = await self._normalizer.send_prompt_async(
                message=request, target=self.target, conversation_id=self.conversation_id
            )
            self._prefix.append(self._message_key(new_messages[-1]))
        if response.api_role != "assistant" or any(
            piece.has_error() or piece.is_truncated or piece.is_simulated for piece in response.message_pieces
        ):
            raise ValueError("The target did not return a complete authentic provider response.")
        if len(response.model_dump_json().encode("utf-8")) > self.max_response_bytes:
            raise RuntimeError("The actual provider response exceeds the configured offline evidence byte limit.")
        usage = TokenUsage.from_metadata(response.get_piece().prompt_metadata)
        self._last_usage = usage
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(usage, name) if usage is not None else None
            if value is None:
                self._missing_usage.add(name)
            else:
                self.usage[name] = self.usage.get(name, 0) + value
        await self.artifacts.append_async(
            event="native_generation_response",
            data={"ordinal": self.requests, "message": response.model_dump(mode="json")},
        )
        return response

    async def _execute_tool_async(self, state: TaskState) -> None:
        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, execute_tools

        self._check_dispatch(state=state, provider=False)
        message = state.messages[-1]
        if not isinstance(message, ChatMessageAssistant) or not message.tool_calls or len(message.tool_calls) != 1:
            raise ValueError("The strict bridge requires exactly one sequential tool call per response.")
        call = message.tool_calls[0]
        if call.id in self._call_ids:
            raise RuntimeError("Repeated provider tool identity cannot be executed twice.")
        self._call_ids.add(call.id)
        record: dict[str, Any] = {
            "provider_call_id": call.id,
            "dispatch_id": str(uuid4()),
            "name": call.function,
            "arguments": call.arguments,
            "arguments_json": self._raw_arguments[call.id],
            "status": "prepared",
        }
        self.tool_calls.append(record)
        await self.artifacts.append_async(event="native_tool_dispatch", data=record)
        try:
            self._check_dispatch(state=state, provider=False)
            self._tool_dispatches += 1
            record["status"] = "dispatching"
            result = await execute_tools(
                state.messages, self._observed_tools(state=state, record=record), max_output=self.max_tool_output_bytes
            )
            if result.output is not None or len(result.messages) != 1:
                raise ValueError("Native tool handoffs or extra conversation messages are unsupported.")
            output = result.messages[0]
            if (
                not isinstance(output, ChatMessageTool)
                or output.tool_call_id != call.id
                or not isinstance(output.content, str)
            ):
                raise ValueError("Native tool result must be a correlated string message.")
            state.messages.append(output)
            record.update(status="returned", result=output.model_dump(mode="json"))
            acquired = record.get("acquired_feedback")
            if output.error is None and isinstance(acquired, str) and output.content != acquired:
                record["framework_result"] = output.model_dump(mode="json")
                self._retain_acquired_feedback(state=state, record=record)
                raise InspectNativeLimitError("The native framework changed acquired feedback; it was not forwarded.")
            visible = f"Error: {output.error.message}" if output.error else output.content
            if len(visible.encode("utf-8")) > self.max_tool_output_bytes:
                raise InspectNativeLimitError(
                    "Native tool feedback cannot be delivered within the explicit byte bound."
                )
            if output.error and output.error.type == "cancelled":
                record.update(
                    status="cancelled",
                    terminal_diagnostic={
                        "source": "inspect.execute_tools",
                        "message": output.error.message,
                        "forwarded_to_provider": False,
                    },
                )
                self._retain_acquired_feedback(state=state, record=record)
                raise RuntimeError("Native tool execution was cancelled; no further dispatch.")
            await self.artifacts.append_async(event="native_tool_return", data=record)
        except _NativeBudgetStopError:
            record["status"] = "not_dispatched"
            raise
        except BaseException as error:
            self._retain_acquired_feedback(state=state, record=record)
            record.update(status="error", error_type=type(error).__name__, error=str(error))
            try:
                await self.artifacts.append_async(event="native_tool_error", data=record)
            except BaseException as retention_error:
                primary = self._callback_cancellation or error
                primary.add_note(f"Tool evidence retention failed: {type(retention_error).__name__}: {retention_error}")
                logger.warning("Could not retain native tool error evidence: %s", retention_error)
            raise

    def _observed_tools(self, *, state: TaskState, record: dict[str, Any]) -> list[ToolDef]:
        from inspect_ai.tool import ToolDef

        definitions = [ToolDef(tool) for tool in state.tools]
        observed: list[ToolDef] = []
        for definition in definitions:
            original = definition.tool

            def observe(callback: Callable[..., Any]) -> Callable[..., Awaitable[str]]:
                @wraps(callback)
                async def invoke_async(**arguments: Any) -> str:
                    self._check_dispatch(state=state, provider=False, reserved_tool=True)
                    try:
                        result = await callback(**arguments)
                    except asyncio.CancelledError as error:
                        self._callback_cancellation = error
                        raise
                    if not isinstance(result, str):
                        raise ValueError("The strict native task contract requires string tool results.")
                    record["acquired_feedback"] = result
                    if self.on_tool_return is not None:
                        self.on_tool_return()
                    length = len(result.encode("utf-8"))
                    if length > self.max_tool_output_bytes:
                        record["feedback_limit"] = {
                            "source": "native_tool_callback",
                            "bytes": length,
                            "limit_bytes": self.max_tool_output_bytes,
                            "forwarded_to_provider": False,
                        }
                        raise InspectNativeLimitError("Exact native feedback exceeds its configured UTF-8 byte bound.")
                    return result

                return invoke_async

            observed.append(
                ToolDef(
                    tool=observe(original),
                    name=definition.name,
                    description=definition.description,
                    parameters=definition.parameters,
                    parallel=definition.parallel,
                    viewer=definition.viewer,
                    options=definition.options,
                )
            )
        return observed

    @staticmethod
    def _retain_acquired_feedback(*, state: TaskState, record: dict[str, Any]) -> None:
        from inspect_ai.model import ChatMessageTool

        feedback = record.get("acquired_feedback")
        if not isinstance(feedback, str):
            return
        call_id = record["provider_call_id"]
        existing = next(
            (
                message
                for message in state.messages
                if isinstance(message, ChatMessageTool) and message.tool_call_id == call_id
            ),
            None,
        )
        if existing is not None and existing.error is None and existing.content == feedback:
            record["result"] = existing.model_dump(mode="json")
            return
        acquired = ChatMessageTool(
            content=feedback,
            tool_call_id=call_id,
            function=record["name"],
            metadata={"source": "native_tool_callback", "acquired_before_interruption": True},
        )
        if existing is not None:
            record.setdefault(
                "terminal_diagnostic",
                {
                    "source": "inspect.execute_tools",
                    "native_result": existing.model_dump(mode="json"),
                    "forwarded_to_provider": False,
                },
            )
            state.messages[state.messages.index(existing)] = acquired
        else:
            state.messages.append(acquired)
        record["result"] = acquired.model_dump(mode="json")

    def _new_messages(self, state: TaskState) -> list[ChatMessage]:
        if [self._message_key(message) for message in state.messages[: len(self._prefix)]] != self._prefix:
            raise ValueError(
                "The native task changed already-retained history; only authentic appended turns are supported."
            )
        return list(state.messages[len(self._prefix) :])

    def _check_memory(self) -> None:
        if CentralMemory.get_memory_instance() is not self._normalizer.memory:
            raise RuntimeError("CentralMemory changed before native provider/tool dispatch.")

    def pending_message_ids(self) -> list[str]:
        """
        Identify unjoined writes without claiming their database outcome.

        Returns:
            list[str]: Native message identities whose persistence tasks have not finished.
        """
        return [str(json.loads(key).get("id")) for key, task in self._persistence.items() if not task.done()]

    async def _retain_pending_async(self, state: TaskState) -> None:
        for native in self._new_messages(state):
            await self._persist_native_async(native)

    async def _persist_native_async(self, native: ChatMessage) -> None:
        key = self._message_key(native)
        if key in self._prefix:
            return
        persistence = self._persistence.get(key)
        if persistence is None:
            message = await self._native_message_async(native)
            persistence = asyncio.create_task(self._persist_message_async(message))
            self._persistence[key] = persistence
        await asyncio.shield(persistence)
        self._prefix.append(key)

    @staticmethod
    async def _persist_message_async(message: Message) -> None:
        for piece in message.message_pieces:
            await set_message_piece_sha256_async(piece)
        await asyncio.to_thread(CentralMemory.get_memory_instance().add_message_to_memory, request=message)

    async def _native_message_async(self, native: ChatMessage) -> Message:
        from inspect_ai.model import ChatMessageTool

        if not isinstance(native.content, str):
            raise ValueError("This strict bridge accepts native text messages only.")
        metadata: dict[str, Any] = {"inspect_message_id": native.id}
        if isinstance(native, ChatMessageTool):
            if native.tool_call_id is None:
                raise ValueError("Native tool feedback has no provider call identity.")
            # Inspect's OpenAI wire representation uses this prefix only for ToolError.
            feedback = f"Error: {native.error.message}" if native.error else native.content
            value = json.dumps(
                {"type": "function_call_output", "call_id": native.tool_call_id, "output": feedback},
                separators=(",", ":"),
            )
            metadata["inspect_tool_result"] = native.model_dump(mode="json")
            return MessagePiece(
                role="tool",
                conversation_id=self.conversation_id,
                original_value=value,
                original_value_data_type="function_call_output",
                prompt_metadata=metadata,
            ).to_message()
        if native.role not in ("system", "user"):
            raise ValueError("Native-added assistant messages require a separate provenance contract.")
        return MessagePiece(
            role=native.role,
            conversation_id=self.conversation_id,
            original_value=native.content,
            prompt_metadata=metadata,
        ).to_message()

    def _assistant_message(self, response: Message) -> ChatMessageAssistant:
        from inspect_ai.model import ChatMessageAssistant
        from inspect_ai.tool import ToolCall

        texts: list[str] = []
        calls: list[ToolCall] = []
        for piece in response.message_pieces:
            if piece.converted_value_data_type == "text":
                texts.append(piece.converted_value)
            elif piece.converted_value_data_type == "function_call":
                value = json.loads(piece.converted_value)
                self._raw_arguments[value["call_id"]] = value["arguments"]
                calls.append(
                    ToolCall(id=value["call_id"], function=value["name"], arguments=json.loads(value["arguments"]))
                )
            elif piece.converted_value_data_type != "reasoning":
                raise ValueError("The strict native bridge supports text and function calls, not other model tools.")
        if len(calls) > 1 or (not calls and not any(texts)):
            raise ValueError("Expected one sequential tool call or nonempty actual assistant text.")
        return ChatMessageAssistant(
            content="\n".join(texts),
            tool_calls=calls or None,
            model=self.model_name,
            metadata={"pyrit_message_piece_ids": [str(piece.id) for piece in response.message_pieces]},
        )

    def _set_output(self, *, state: TaskState, assistant: ChatMessageAssistant) -> None:
        from inspect_ai.model import ChatCompletionChoice, ModelOutput, ModelUsage

        usage = self._last_usage
        native_usage = None
        if (
            usage is not None
            and usage.input_tokens is not None
            and usage.output_tokens is not None
            and usage.total_tokens is not None
        ):
            native_usage = ModelUsage(
                input_tokens=usage.input_tokens - (usage.cached_tokens or 0),
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                input_tokens_cache_read=usage.cached_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            )
        state.output = ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(message=assistant, stop_reason="tool_calls" if assistant.tool_calls else "stop")
            ],
            usage=native_usage,
        )

    @staticmethod
    def _message_key(message: ChatMessage) -> str:
        return message.model_dump_json()
