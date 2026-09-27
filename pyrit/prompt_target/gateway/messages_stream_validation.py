# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# DOC501 treats the GatewayError factory as a separate exception.
# ruff: noqa: DOC501

"""Stateful validation of original Anthropic Messages SSE frames and tool ordering."""

from dataclasses import dataclass
from typing import Any

from pyrit.prompt_target.gateway.json_utility import strict_json_loads
from pyrit.prompt_target.gateway.messages_contract import MessagesCoverage
from pyrit.prompt_target.gateway.messages_validation import MessagesValidator
from pyrit.prompt_target.gateway.responses_contract import GatewayError


def _invalid_stream(*, message: str) -> GatewayError:
    return GatewayError(status_code=502, code="invalid_backend_response", message=message)


@dataclass(frozen=True)
class ParsedMessagesEvent:
    """One original SSE event, possibly a comment-only keep-alive."""

    name: str | None
    data: dict[str, Any] | None
    comment: str | None

    @property
    def credential_scan_data(self) -> dict[str, Any]:
        """All decoded strings that could include an echoed host credential."""
        if self.data is None:
            return {"comment": self.comment}
        return {"event": self.data, "comment": self.comment}


def parse_messages_sse_frame(*, frame: bytes) -> ParsedMessagesEvent:
    """
    Decode an Anthropic SSE frame without changing the bytes returned to the caller.

    Args:
        frame (bytes): One complete original provider SSE frame.

    Returns:
        ParsedMessagesEvent: Event name, JSON payload, and optional keep-alive comment.

    Raises:
        GatewayError: If the provider sent malformed SSE or duplicate fields.
    """
    try:
        normalized = frame.decode("utf-8").replace("\r\n", "\n")
        if not normalized.endswith("\n\n") or "\r" in normalized:
            raise ValueError("SSE frame is incomplete")
        lines = normalized[:-2].split("\n")
        events = [line[6:].strip() for line in lines if line.startswith("event:")]
        data_lines = [line[5:].lstrip(" ") for line in lines if line.startswith("data:")]
        comments = [line[1:].lstrip(" ") for line in lines if line.startswith(":")]
        if (
            len(events) > 1
            or any(not line.startswith(("event:", "data:", ":")) for line in lines)
            or (not events and bool(data_lines))
        ):
            raise ValueError("Unsupported SSE field")
        if not events and comments and not data_lines:
            return ParsedMessagesEvent(name=None, data=None, comment="\n".join(comments))
        if len(events) != 1 or not events[0] or not data_lines:
            raise ValueError("SSE event name or JSON data missing")
        data = strict_json_loads(value="\n".join(data_lines))
        if not isinstance(data, dict) or data.get("type") != events[0]:
            raise ValueError("SSE event and payload type differ")
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise _invalid_stream(message="Model backend yielded invalid Anthropic SSE") from exc
    return ParsedMessagesEvent(name=events[0], data=data, comment="\n".join(comments) if comments else None)


class MessagesStreamValidator:
    """Check a real message_start -> block sequence -> message_delta -> message_stop."""

    _STOP_REASONS = (
        "end_turn",
        "tool_use",
        "max_tokens",
        "stop_sequence",
        "pause_turn",
        "refusal",
        "model_context_window_exceeded",
    )

    def __init__(
        self,
        *,
        validator: MessagesValidator,
        advertised_tools: frozenset[str],
        max_tokens: int,
        max_bytes: int,
    ) -> None:
        """Record one request's tool identities, output-token reservation, and byte cap."""
        self._validator = validator
        self._advertised_tools = advertised_tools
        self._max_tokens = max_tokens
        self._max_bytes = max_bytes
        self._received_bytes = 0
        self._started = False
        self._open_index: int | None = None
        self._open_kind: str | None = None
        self._next_index = 0
        self._tool_input: list[str] = []
        self._tool_ids: set[str] = set()
        self._tool_seen = False
        self._thinking_signed = False
        self._saw_final_usage = False
        self._last_tokens = 0
        self._stop_reason: str | None = None
        self.terminal = False
        self.failed = False

    def accept(self, *, frame: bytes) -> frozenset[MessagesCoverage]:
        """
        Validate and account for one unchanged provider SSE frame.

        Args:
            frame (bytes): Complete original provider SSE event.

        Returns:
            frozenset[MessagesCoverage]: Genuine features present on this event.

        Raises:
            GatewayError: If ordering, usage, tools, event shape, or bytes are unsafe.
        """
        if not isinstance(frame, bytes):
            raise _invalid_stream(message="Model backend SSE frames must be bytes")
        self._received_bytes += len(frame)
        if self._received_bytes > self._max_bytes:
            raise GatewayError(status_code=502, code="response_too_large", message="Model response exceeded byte limit")
        parsed = parse_messages_sse_frame(frame=frame)
        if self.terminal:
            raise _invalid_stream(message="Model backend emitted an event after stream termination")
        if parsed.name is None or parsed.name == "ping":
            return frozenset({MessagesCoverage.STREAMING, MessagesCoverage.PING})
        if parsed.data is None:
            raise _invalid_stream(message="Model backend SSE event has no data")
        coverage = set(self._handle_event(name=parsed.name, data=parsed.data))
        coverage.add(MessagesCoverage.STREAMING)
        return frozenset(coverage)

    def _handle_event(self, *, name: str, data: dict[str, Any]) -> frozenset[MessagesCoverage]:
        if name == "message_start":
            if self._started:
                raise _invalid_stream(message="Model backend sent duplicate message_start")
            coverage = self._validator.validate_provider_message(
                value=data.get("message"),
                max_tokens=self._max_tokens,
                advertised_tools=self._advertised_tools,
                streaming_start=True,
            )
            message = data["message"]
            self._last_tokens = message["usage"]["output_tokens"]
            self._started = True
            return coverage
        if name == "error":
            self._validator.validate_provider_error(value=data)
            self.terminal = self.failed = True
            return frozenset({MessagesCoverage.FAILED})
        if not self._started:
            raise _invalid_stream(message="Model backend stream must begin with message_start")
        if name == "content_block_start":
            return self._start_block(data=data)
        if name == "content_block_delta":
            return self._delta(data=data)
        if name == "content_block_stop":
            return self._stop_block(data=data)
        if name == "message_delta":
            return self._message_delta(data=data)
        if name == "message_stop":
            if self._open_index is not None or self._stop_reason is None or not self._saw_final_usage:
                raise _invalid_stream(message="Model backend stopped before closing content and reporting usage")
            self.terminal = True
            return frozenset({MessagesCoverage.COMPLETED})
        raise _invalid_stream(message="Model backend returned an unsupported Anthropic SSE event")

    def _start_block(self, *, data: dict[str, Any]) -> frozenset[MessagesCoverage]:
        index = data.get("index")
        if (
            self._stop_reason is not None
            or self._open_index is not None
            or type(index) is not int
            or index != self._next_index
        ):
            raise _invalid_stream(message="Model backend content block started out of order")
        coverage = self._validator.validate_provider_block(
            value=data.get("content_block"), advertised_tools=self._advertised_tools, starting=True
        )
        block = data["content_block"]
        self._open_kind = block["type"]
        self._open_index = index
        self._next_index += 1
        self._tool_input = []
        self._thinking_signed = False
        if self._open_kind == "tool_use":
            if block["id"] in self._tool_ids:
                raise _invalid_stream(message="Model backend reused a tool_use id")
            self._tool_ids.add(block["id"])
            self._tool_seen = True
        return coverage

    def _delta(self, *, data: dict[str, Any]) -> frozenset[MessagesCoverage]:
        if self._open_index is None or type(data.get("index")) is not int or data["index"] != self._open_index:
            raise _invalid_stream(message="Model backend content delta has no open block")
        delta = data.get("delta")
        if not isinstance(delta, dict):
            raise _invalid_stream(message="Model backend content delta must be an object")
        kind = delta.get("type")
        if self._open_kind == "text" and kind == "text_delta" and isinstance(delta.get("text"), str):
            return frozenset({MessagesCoverage.TEXT})
        if self._open_kind == "tool_use" and kind == "input_json_delta" and isinstance(delta.get("partial_json"), str):
            self._tool_input.append(delta["partial_json"])
            return frozenset({MessagesCoverage.TOOL_USE})
        if self._open_kind == "thinking" and kind == "thinking_delta" and isinstance(delta.get("thinking"), str):
            return frozenset({MessagesCoverage.THINKING})
        if self._open_kind == "thinking" and kind == "signature_delta" and isinstance(delta.get("signature"), str):
            self._thinking_signed = True
            return frozenset({MessagesCoverage.THINKING})
        raise _invalid_stream(message="Model backend sent a delta for the wrong content block")

    def _stop_block(self, *, data: dict[str, Any]) -> frozenset[MessagesCoverage]:
        if self._open_index is None or type(data.get("index")) is not int or data["index"] != self._open_index:
            raise _invalid_stream(message="Model backend stopped a content block that was not open")
        kind = self._open_kind
        if kind == "tool_use" and "".join(self._tool_input):
            try:
                payload = strict_json_loads(value="".join(self._tool_input))
            except (ValueError, RecursionError) as exc:
                raise _invalid_stream(message="Model backend tool input JSON is incomplete") from exc
            if not isinstance(payload, dict):
                raise _invalid_stream(message="Model backend tool input must be an object")
        if kind == "thinking" and not self._thinking_signed:
            raise _invalid_stream(message="Model backend stopped unsigned thinking")
        self._open_index = None
        self._open_kind = None
        self._tool_input = []
        return frozenset({MessagesCoverage.TOOL_USE}) if kind == "tool_use" else frozenset[MessagesCoverage]()

    def _message_delta(self, *, data: dict[str, Any]) -> frozenset[MessagesCoverage]:
        if self._open_index is not None or not isinstance(data.get("delta"), dict):
            raise _invalid_stream(message="Model backend message_delta occurred before closing content")
        delta = data["delta"]
        reason = delta.get("stop_reason")
        if reason is not None:
            if reason not in self._STOP_REASONS or (self._stop_reason and reason != self._stop_reason):
                raise _invalid_stream(message="Model backend returned an unsupported stop reason")
            if reason == "tool_use" and not self._tool_seen:
                raise _invalid_stream(message="Model backend stopped for tool use without a tool_use block")
            self._stop_reason = reason
        if "usage" in data:
            usage = data["usage"]
            if not isinstance(usage, dict):
                raise _invalid_stream(message="Model backend message_delta usage must be an object")
            tokens = usage.get("output_tokens")
            if type(tokens) is not int or not self._last_tokens <= tokens <= self._max_tokens:
                raise _invalid_stream(message="Model backend output-token usage exceeded the reservation")
            self._last_tokens = tokens
            self._saw_final_usage = True
        return frozenset[MessagesCoverage]()
