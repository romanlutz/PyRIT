# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol

from pyrit.models import ComponentIdentifier, Message, MessagePiece
from pyrit.models.native_cyber import (
    NativeAgentCapabilities,
    NativeAgentEvent,
    NativeAgentEvidence,
    NativeToolRequest,
    NativeToolTrace,
)
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_capabilities import TargetCapabilities
from pyrit.prompt_target.common.target_configuration import TargetConfiguration

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import JsonValue

logger = logging.getLogger(__name__)


class NativeAgentSession(Protocol):
    """A retained agent owned by an external environment lease."""

    session_id: str
    environment_id: str
    capabilities: NativeAgentCapabilities
    simulated: bool

    async def send_async(self, *, prompt: str, timeout_seconds: float) -> tuple[NativeAgentEvent, ...]:
        """Send one operator instruction and wait for a quiescent native turn."""
        ...

    def evidence(self) -> NativeAgentEvidence:
        """Return a detached snapshot of actual agent events."""
        ...

    async def quiesce_async(self) -> None:
        """Stop the agent before original grading, without removing its task environment."""
        ...


class CopilotSdkEvent(Protocol):
    """Public SDK event serialization surface."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize a native SDK event without rewriting its fields."""
        ...


class CopilotSdkSession(Protocol):
    """The pinned public SDK subset; no user-defined tool execution callback is accepted."""

    session_id: str

    def on(self, handler: Callable[[Any], None]) -> Callable[[], None]:
        """Subscribe to the SDK's generated event union."""
        ...

    async def send_and_wait(self, prompt: str, *, timeout: float) -> object:  # pyrit-async-suffix-exempt
        """Wait for one actual native session-idle boundary."""
        ...

    async def disconnect(self) -> None:  # pyrit-async-suffix-exempt
        """Detach the SDK session; caller environment cleanup remains mandatory."""
        ...


class CopilotSdkAgentSession:
    """Adapt a caller-qualified external Copilot CLI session, without creating auth or host tools."""

    def __init__(
        self,
        *,
        session: CopilotSdkSession,
        environment_id: str,
        simulated: bool,
        capabilities: NativeAgentCapabilities,
        provenance: dict[str, JsonValue],
        max_events: int = 10000,
        max_event_bytes: int = 1_048_576,
    ) -> None:
        """
        Bind an already-created SDK session; qualification and environment lifetime belong to the caller.

        Raises:
            ValueError: If event retention bounds are invalid.
        """
        self._session = session
        self.session_id = session.session_id
        self.environment_id = environment_id
        self.simulated = simulated
        self.capabilities = capabilities
        self._provenance = provenance
        self._events: list[NativeAgentEvent] = []
        self._tools: dict[str, NativeToolTrace] = {}
        self._requests: dict[str, NativeToolRequest] = {}
        self._gaps: list[str] = []
        self._ids: set[str] = set()
        self._idle = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._event_lock = threading.RLock()
        self._max_events = max_events
        self._max_event_bytes = max_event_bytes
        if max_events <= 0 or max_event_bytes <= 0:
            raise ValueError("Native event retention limits must be positive.")
        self._unsubscribe = session.on(self._receive)

    async def send_async(self, *, prompt: str, timeout_seconds: float) -> tuple[NativeAgentEvent, ...]:
        """
        Invoke the existing CLI agent and retain its native events, including failed turns.

        Returns:
            tuple[NativeAgentEvent, ...]: Actual new event records.

        Raises:
            RuntimeError: If the session is closed or lacks a verified idle boundary.
        """
        async with self._lock:
            if self._closed:
                raise RuntimeError("The native agent session is closed.")
            start = len(self._events)
            self._idle = False
            async with asyncio.timeout(timeout_seconds):
                await self._session.send_and_wait(prompt, timeout=timeout_seconds)
            if not self.evidence().idle:
                raise RuntimeError("The native CLI returned without an observed root session.idle event.")
            return tuple(self._events[start:])

    def evidence(self) -> NativeAgentEvidence:
        """
        Snapshot native event coverage and correlated tool observations.

        Returns:
            NativeAgentEvidence: No inferred or synthetic tool receipts.
        """
        with self._event_lock:
            gaps = list(self._gaps)
            if any(tool.status == "running" for tool in self._tools.values()):
                gaps.append("One or more observed native tool calls have no completion event.")
            if set(self._requests) - set(self._tools):
                gaps.append("One or more model-requested tool calls have no observed native execution.")
            if set(self._tools) - set(self._requests):
                gaps.append("One or more native executions have no retained model tool request.")
            if not self._idle:
                gaps.append("No quiescent native session boundary is currently observed.")
            return NativeAgentEvidence(
                session_id=self.session_id,
                environment_id=self.environment_id,
                simulated=self.simulated,
                events=tuple(self._events),
                tools=tuple(self._tools.values()),
                tool_requests=tuple(self._requests.values()),
                idle=self._idle,
                coverage_complete=not gaps,
                gaps=tuple(gaps),
                provenance=self._provenance,
            ).model_copy(deep=True)

    async def quiesce_async(self) -> None:
        """
        Detach only at a known idle boundary; the environment owner then prevents further execution.

        Raises:
            RuntimeError: If the agent is not known idle.
        """
        if self._closed:
            return
        if not self._idle or self._lock.locked():
            raise RuntimeError("Cannot claim agent termination while native work may still be running.")
        await self._session.disconnect()
        self._unsubscribe()
        self._closed = True

    @staticmethod
    def docker_stdio_arguments(*, container_id: str, cli_path: str) -> tuple[str, ...]:
        """
        Build the public SDK ``RuntimeConnection.for_stdio(path="docker", args=...)`` prefix.

        The SDK appends its headless/stdio flags after this prefix. This creates
        no container, forwards no environment, supplies no auth and opens no port.

        Returns:
            tuple[str, ...]: An argv prefix naming only one caller-owned container.

        Raises:
            ValueError: If the container ID or Linux executable path is not explicit.
        """
        path = PurePosixPath(cli_path)
        if not re.fullmatch(r"[0-9a-f]{64}", container_id):
            raise ValueError("Docker stdio requires the full owned container ID.")
        if not path.is_absolute() or ".." in path.parts or "\x00" in cli_path or "\\" in cli_path:
            raise ValueError("The pinned CLI executable requires an absolute Linux path.")
        return ("exec", "-i", container_id, str(path))

    def _receive(self, event: CopilotSdkEvent) -> None:
        with self._event_lock:
            self._receive_locked(event)

    def _receive_locked(self, event: CopilotSdkEvent) -> None:
        try:
            raw = event.to_dict()
            if raw.get("sessionId", self.session_id) != self.session_id:
                raise ValueError("Native event belongs to a different session.")
            encoded = json.dumps(raw, ensure_ascii=True, allow_nan=False)
            if len(self._events) >= self._max_events or len(encoded.encode("utf-8")) > self._max_event_bytes:
                raise ValueError("Native event retention limit exceeded; full coverage is unavailable.")
            event_id, event_type = raw.get("id"), raw.get("type")
            if not isinstance(event_id, str) or not event_id or not isinstance(event_type, str):
                raise ValueError("Native events require actual IDs and types.")
            if event_id in self._ids:
                raise ValueError("Native event ID was repeated.")
            self._ids.add(event_id)
            retained = NativeAgentEvent(
                sequence=len(self._events) + 1,
                event_id=event_id,
                session_id=self.session_id,
                event_type=event_type,
                payload=json.loads(encoded),
            )
            self._events.append(retained)
            self._correlate(retained)
        except (ValueError, TypeError, KeyError) as error:
            self._gaps.append(str(error))
            logger.error("Native agent evidence is incomplete: %s", error)

    def _correlate(self, event: NativeAgentEvent) -> None:
        raw = event.payload
        data = raw.get("data")
        if not isinstance(data, dict):
            raise ValueError("Native event data must be a structured object.")
        if event.event_type == "assistant.message":
            requests = data.get("toolRequests", []) or []
            if not isinstance(requests, list):
                raise ValueError("Native tool requests must be a structured list.")
            for request in requests:
                if not isinstance(request, dict):
                    raise ValueError("Native tool request must be a structured object.")
                call_id, name = request.get("toolCallId"), request.get("name")
                if not isinstance(call_id, str) or not isinstance(name, str) or call_id in self._requests:
                    raise ValueError("Native assistant tool request has missing or reused identity.")
                self._requests[call_id] = NativeToolRequest(
                    call_id=call_id, name=name, arguments=request.get("arguments"), request_sequence=event.sequence
                )
        elif event.event_type == "session.idle" and not raw.get("agentId"):
            self._idle = data.get("aborted") is not True
        elif event.event_type == "session.error":
            self._gaps.append("Native session.error was observed.")
        elif event.event_type == "tool.execution_start":
            call_id, name = data.get("toolCallId"), data.get("toolName")
            if not isinstance(call_id, str) or not isinstance(name, str) or call_id in self._tools:
                raise ValueError("Native tool start has missing or repeated identity.")
            self._tools[call_id] = NativeToolTrace(
                call_id=call_id,
                name=name,
                arguments=data.get("arguments"),
                start_sequence=event.sequence,
                request_sequence=self._requests[call_id].request_sequence if call_id in self._requests else None,
            )
            request = self._requests.get(call_id)
            if request is None:
                raise ValueError("Native tool execution started before its model tool request was observed.")
            if request.name != name or request.arguments != data.get("arguments"):
                raise ValueError("Native tool execution differs from its model-requested name or arguments.")
        elif event.event_type == "tool.execution_complete":
            call_id, success = data.get("toolCallId"), data.get("success")
            if not isinstance(call_id, str) or type(success) is not bool:
                raise ValueError("Native tool completion lacks ID or success status.")
            previous = self._tools.get(call_id)
            if previous is None or previous.completion_sequence is not None:
                raise ValueError("Native tool completion has no unique start.")
            result = data.get("result")
            model_visible_output = result.get("content") if isinstance(result, dict) else None
            detailed_output = result.get("detailedContent") if isinstance(result, dict) else None
            if (model_visible_output is not None and not isinstance(model_visible_output, str)) or (
                detailed_output is not None and not isinstance(detailed_output, str)
            ):
                raise ValueError("Native tool output text must be a string or absent.")
            self._tools[call_id] = NativeToolTrace(
                call_id=call_id,
                name=previous.name,
                arguments=previous.arguments,
                start_sequence=previous.start_sequence,
                completion_sequence=event.sequence,
                success=success,
                result=data.get("result"),
                error=data.get("error"),
                status="succeeded" if success else "failed",
                request_sequence=previous.request_sequence,
                model_visible_output=model_visible_output,
                detailed_output=detailed_output,
            )
            if not isinstance(result, dict) and success:
                self._gaps.append("Native tool success has no retained result payload.")


class NativeAgentTarget(PromptTarget):
    """Send prepared prompts to a lease-owned native CLI session, never to a host shell."""

    def __init__(self, *, session: NativeAgentSession, turn_timeout_seconds: float = 60) -> None:
        """
        Configure the public target around an existing, qualified native session.

        Raises:
            ValueError: If the per-turn deadline is nonpositive.
        """
        if turn_timeout_seconds <= 0:
            raise ValueError("A native target requires a positive per-turn timeout.")
        super().__init__(
            custom_configuration=TargetConfiguration(
                capabilities=TargetCapabilities(
                    supports_multi_turn=session.capabilities.retained_session,
                    supports_editable_history=False,
                    supports_multi_message_pieces=True,
                    supports_system_prompt=False,
                    input_modalities=frozenset(
                        {
                            frozenset({"text"}),
                            frozenset({"function_call"}),
                            frozenset({"function_call_output"}),
                        }
                    ),
                )
            )
        )
        self.session = session
        self._turn_timeout_seconds = turn_timeout_seconds
        self._conversation_id: str | None = None

    @property
    def conversation_id(self) -> str | None:
        """The actual conversation accepted for a send, if one was started."""
        return self._conversation_id

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(
            params={
                "native_session_id": self.session.session_id,
                "environment_id": self.session.environment_id,
                "simulated": self.session.simulated,
                "adapter": "native_agent_session",
            }
        )

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        request = normalized_conversation[-1]
        if request.api_role != "user":
            raise ValueError("Native agent turns require prepared user instructions, not replayed tool execution.")
        if self._conversation_id is not None and request.conversation_id != self._conversation_id:
            raise ValueError("A native session cannot be silently reused for another conversation or rerun.")
        self._conversation_id = request.conversation_id
        async with asyncio.timeout(self._turn_timeout_seconds):
            events = await self.session.send_async(
                prompt="\n".join(request.get_values()), timeout_seconds=self._turn_timeout_seconds
            )
        responses: list[Message] = []
        for event in events:
            data = event.payload.get("data")
            if not isinstance(data, dict):
                continue
            content = data.get("content")
            if event.event_type == "assistant.message" and isinstance(content, str) and content:
                responses.append(
                    MessagePiece(
                        role="assistant",
                        conversation_id=request.conversation_id,
                        original_value=content,
                        prompt_metadata={"native_event_id": event.event_id, "native_session_id": event.session_id},
                    ).to_message()
                )
            elif event.event_type in {"tool.execution_start", "tool.execution_complete"}:
                responses.append(
                    MessagePiece(
                        role="assistant" if event.event_type == "tool.execution_start" else "tool",
                        conversation_id=request.conversation_id,
                        original_value_data_type="function_call"
                        if event.event_type == "tool.execution_start"
                        else "function_call_output",
                        original_value=json.dumps(event.payload, separators=(",", ":")),
                        prompt_metadata={"native_event_id": event.event_id, "native_session_id": event.session_id},
                    ).to_message()
                )
        # Empty list remains a write-only result, never a manufactured assistant receipt.
        return responses
