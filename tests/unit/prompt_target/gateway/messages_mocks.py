# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Inert Anthropic Messages fixtures shared by ASGI and HTTP transport tests."""

import json
from collections.abc import AsyncGenerator
from typing import Any

from pyrit.prompt_target.gateway.messages_contract import (
    MessagesCapabilities,
    MessagesRequest,
    MessagesResponse,
    MessagesStream,
)
from pyrit.prompt_target.gateway.responses_contract import GatewayLimits, GatewayRoute

ROUTE = GatewayRoute(run_id="claude-offline-run", model="claude-offline-model", guest_token="guest-only-" + "x" * 32)
HOST_KEY = "host-only-" + "y" * 32
LIMITS = GatewayLimits(max_request_bytes=8_192, max_response_bytes=16_384, max_output_tokens_per_request=64)
CAPABILITIES = MessagesCapabilities(
    streaming=True,
    tool_use=True,
    thinking=True,
    prompt_caching=True,
    effort=True,
    allowed_beta_values=frozenset({"verified-tool-beta-2026-09-01"}),
)
GUEST_HEADERS = {
    "Authorization": f"Bearer {ROUTE.guest_token}",
    "X-PyRIT-Run-ID": ROUTE.run_id,
    "anthropic-version": "2023-06-01",
}
TOOL = {
    "name": "shell_command",
    "description": "Return offline text from a sandboxed CLI tool",
    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
}
TOOL_CALL = {"type": "tool_use", "id": "toolu-offline-1", "name": "shell_command", "input": {"command": "OFFLINE"}}
TOOL_RESULT = {"type": "tool_result", "tool_use_id": "toolu-offline-1", "content": "exact\nOFFLINE result"}


def request_body(*, streaming: bool = False, tools: bool = False) -> dict[str, Any]:
    """Construct a valid minimal Anthropic request with an explicit token limit."""
    body: dict[str, Any] = {
        "model": ROUTE.model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "OFFLINE"}],
    }
    if streaming:
        body["stream"] = True
    if tools:
        body["tools"] = [TOOL]
    return body


def message(*, content: list[dict[str, Any]] | None = None, stop_reason: str = "end_turn") -> dict[str, Any]:
    """Build a synthetic provider Message with real Anthropic shape and usage."""
    return {
        "type": "message",
        "id": "msg_offline_1",
        "role": "assistant",
        "model": ROUTE.model,
        "content": content if content is not None else [{"type": "text", "text": "OFFLINE reply"}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 9, "output_tokens": 3, "cache_read_input_tokens": 0},
    }


def response_bytes(*, content: list[dict[str, Any]] | None = None, stop_reason: str = "end_turn") -> bytes:
    """Serialize only the synthetic provider's genuine Message body."""
    return json.dumps(message(content=content, stop_reason=stop_reason), separators=(",", ":")).encode()


def event(*, name: str, **fields: Any) -> bytes:
    """Render a complete raw Anthropic SSE frame."""
    data = json.dumps({"type": name, **fields}, separators=(",", ":"))
    return f"event: {name}\ndata: {data}\n\n".encode()


def text_frames() -> list[bytes]:
    """Return a complete text response with an original comment and ping."""
    start = {**message(content=[]), "stop_reason": None}
    start["usage"] = {"input_tokens": 9, "output_tokens": 1}
    return [
        event(name="message_start", message=start),
        b": provider keep-alive\n\n",
        event(name="content_block_start", index=0, content_block={"type": "text", "text": ""}),
        event(name="ping"),
        event(name="content_block_delta", index=0, delta={"type": "text_delta", "text": "OFFLINE reply"}),
        event(name="content_block_stop", index=0),
        event(
            name="message_delta", delta={"stop_reason": "end_turn", "stop_sequence": None}, usage={"output_tokens": 3}
        ),
        event(name="message_stop"),
    ]


def tool_frames() -> list[bytes]:
    """Return a complete CLI-owned tool_use event sequence."""
    start = {**message(content=[]), "stop_reason": None}
    start["usage"] = {"input_tokens": 9, "output_tokens": 1}
    return [
        event(name="message_start", message=start),
        event(
            name="content_block_start",
            index=0,
            content_block={"type": "tool_use", "id": "toolu-offline-1", "name": "shell_command", "input": {}},
        ),
        event(name="content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": '{"command":'}),
        event(name="content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": '"OFFLINE"}'}),
        event(name="content_block_stop", index=0),
        event(name="message_delta", delta={"stop_reason": "tool_use"}, usage={"output_tokens": 3}),
        event(name="message_stop"),
    ]


class FakeMessagesBackend:
    """Only return caller-supplied wire frames; never open sockets or execute tools."""

    def __init__(
        self,
        *,
        responses: list[MessagesResponse] | None = None,
        streams: list[list[bytes]] | None = None,
        capabilities: MessagesCapabilities = CAPABILITIES,
    ) -> None:
        self.capabilities = capabilities
        self.responses = (
            responses
            if responses is not None
            else [
                MessagesResponse(
                    status_code=200, body=response_bytes(), headers=(("content-type", "application/json"),)
                )
            ]
        )
        self.streams = streams if streams is not None else [text_frames()]
        self.requests: list[MessagesRequest] = []
        self.closed = False

    async def create_message_async(self, *, request: MessagesRequest) -> MessagesResponse:
        """Return one original fake model response."""
        self.requests.append(request)
        return self.responses.pop(0)

    async def open_stream_async(self, *, request: MessagesRequest) -> MessagesStream:
        """Return one closeable fake provider event stream."""
        self.requests.append(request)
        return MessagesStream(
            frames=self._frames_async(frames=self.streams.pop(0)),
            headers=(("content-type", "text/event-stream"),),
            _close=self._close_async,
        )

    async def _frames_async(self, *, frames: list[bytes]) -> AsyncGenerator[bytes, None]:
        for frame in frames:
            yield frame

    async def _close_async(self) -> None:
        self.closed = True
