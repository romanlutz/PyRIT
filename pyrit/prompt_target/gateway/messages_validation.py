# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# DOC501 sees exception factories as distinct types; both return GatewayError.
# ruff: noqa: DOC501

"""Strict Anthropic Messages request and terminal response validation."""

import math
import re
from dataclasses import dataclass
from typing import Any

from pyrit.prompt_target.gateway.messages_contract import MessagesCapabilities, MessagesCoverage
from pyrit.prompt_target.gateway.responses_contract import GatewayError, GatewayLimits, GatewayRoute


def _invalid(*, message: str) -> GatewayError:
    return GatewayError(status_code=400, code="invalid_request", message=message)


def _unsupported(*, message: str) -> GatewayError:
    return GatewayError(status_code=501, code="unsupported_feature", message=message)


def _invalid_provider(*, message: str) -> GatewayError:
    return GatewayError(status_code=502, code="invalid_backend_response", message=message)


def _fields(*, value: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    if not set(value).issubset(allowed):
        raise _unsupported(message=f"Unsupported {where} field")


def _object(*, value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _invalid(message=f"{where} must be a JSON object")
    return value


def _provider_object(*, value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _invalid_provider(message=f"Model backend {where} must be an object")
    return value


def _string(*, value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(message=f"{where} must be a nonempty string")
    return value


def _token_count(*, value: object, where: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise _invalid_provider(message=f"Model backend {where} is outside the reserved token budget")
    return value


@dataclass(frozen=True)
class ValidatedMessages:
    """The accepted, unmodified Anthropic request and observed wire features."""

    body: dict[str, Any]
    max_tokens: int
    streaming: bool
    advertised_tools: frozenset[str]
    coverage: frozenset[MessagesCoverage]


class MessagesValidator:
    """Accept a bounded, documented Anthropic Messages subset without rewriting it."""

    _REQUEST_FIELDS = frozenset(
        {
            "model",
            "max_tokens",
            "messages",
            "stream",
            "system",
            "tools",
            "tool_choice",
            "thinking",
            "output_config",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "metadata",
            "service_tier",
        }
    )
    _MESSAGE_FIELDS = frozenset({"role", "content", "cache_control"})
    _TOOL_FIELDS = frozenset({"type", "name", "description", "input_schema", "cache_control"})
    _BLOCK_FIELDS = {
        "text": frozenset({"type", "text", "cache_control"}),
        "tool_use": frozenset({"type", "id", "name", "input", "cache_control"}),
        "tool_result": frozenset({"type", "tool_use_id", "content", "is_error", "cache_control"}),
        "thinking": frozenset({"type", "thinking", "signature"}),
        "redacted_thinking": frozenset({"type", "data"}),
    }
    _STOP_REASONS = (
        "end_turn",
        "tool_use",
        "max_tokens",
        "stop_sequence",
        "pause_turn",
        "refusal",
        "model_context_window_exceeded",
    )

    def __init__(self, *, route: GatewayRoute, limits: GatewayLimits, capabilities: MessagesCapabilities) -> None:
        """Bind one model, run, budget, and verified upstream capability set."""
        self._route = route
        self._limits = limits
        self._capabilities = capabilities

    def validate_headers(self, *, version: str, beta: str | None) -> None:
        """
        Require the documented Anthropic version and explicitly approved beta values.

        Args:
            version (str): Original ``anthropic-version`` request header.
            beta (str | None): Original comma-separated ``anthropic-beta`` header.

        Raises:
            GatewayError: If a header asks for an unknown protocol capability.
        """
        if version != "2023-06-01":
            raise _unsupported(message="Unsupported anthropic-version header")
        if beta is None:
            return
        if len(beta) > 4096:
            raise _unsupported(message="Unsupported anthropic-beta header")
        values = [part.strip() for part in beta.split(",")]
        if len(values) != len(set(values)) or any(
            not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", part) or "oauth" in part for part in values
        ):
            raise _unsupported(message="Unsupported anthropic-beta header")
        if not set(values).issubset(self._capabilities.allowed_beta_values):
            raise _unsupported(message="Unverified anthropic-beta capability")

    def validate_request(self, *, body: dict[str, Any]) -> ValidatedMessages:
        """
        Validate only features the pinned model can honor, retaining original fields.

        Args:
            body (dict[str, Any]): Strictly decoded original Anthropic JSON body.

        Returns:
            ValidatedMessages: Bounds, tools, and coverage for the unchanged request.

        Raises:
            GatewayError: If a request is malformed, unrouted, or unsupported.
        """
        _fields(value=body, allowed=self._REQUEST_FIELDS, where="Messages request")
        if body.get("model") != self._route.model:
            raise _invalid(message="This model is not routed to this run")
        maximum = body.get("max_tokens")
        if type(maximum) is not int or maximum <= 0:
            raise _invalid(message="max_tokens must be a positive integer")
        if maximum > self._limits.max_output_tokens_per_request:
            raise GatewayError(
                status_code=429,
                code="output_token_limit",
                message="Requested max_tokens exceeds this run's per-request budget",
            )
        streaming = body.get("stream", False)
        if type(streaming) is not bool:
            raise _invalid(message="stream must be a boolean")
        if streaming and not self._capabilities.streaming:
            raise _unsupported(message="Model backend does not support Anthropic SSE streaming")
        coverage = self._validate_messages(value=body.get("messages"))
        if "system" in body:
            coverage.update(self._validate_system(value=body["system"]))
        names, tools_coverage = self._validate_tools(value=body.get("tools", []))
        coverage.update(tools_coverage)
        self._validate_options(body=body, names=names, max_tokens=maximum)
        if "thinking" in body and body["thinking"] != {"type": "disabled"}:
            coverage.add(MessagesCoverage.THINKING)
        if streaming:
            coverage.add(MessagesCoverage.STREAMING)
        return ValidatedMessages(
            body=body,
            max_tokens=maximum,
            streaming=streaming,
            advertised_tools=frozenset(names),
            coverage=frozenset(coverage),
        )

    def _validate_messages(self, *, value: object) -> set[MessagesCoverage]:
        if not isinstance(value, list) or not 0 < len(value) <= 4096:
            raise _invalid(message="messages must be a nonempty bounded array")
        coverage: set[MessagesCoverage] = set()
        for entry in value:
            item = _object(value=entry, where="message")
            _fields(value=item, allowed=self._MESSAGE_FIELDS, where="message")
            role = item.get("role")
            if role not in ("user", "assistant", "system"):
                raise _invalid(message="Unsupported message role")
            if "cache_control" in item:
                coverage.update(self._validate_cache_control(value=item["cache_control"]))
            if "content" not in item:
                raise _invalid(message="Message content is required")
            coverage.update(self._validate_content(value=item["content"], role=role))
        return coverage

    def _validate_content(self, *, value: object, role: str) -> set[MessagesCoverage]:
        if isinstance(value, str):
            return {MessagesCoverage.TEXT}
        if not isinstance(value, list):
            raise _invalid(message="Message content must be text or an array of blocks")
        coverage: set[MessagesCoverage] = set()
        for part in value:
            block = _object(value=part, where="message content block")
            kind = block.get("type")
            if not isinstance(kind, str) or kind not in self._BLOCK_FIELDS:
                raise _unsupported(message="Unsupported Anthropic content block type")
            _fields(value=block, allowed=self._BLOCK_FIELDS[kind], where="message content block")
            if "cache_control" in block:
                coverage.update(self._validate_cache_control(value=block["cache_control"]))
            if kind == "text":
                if not isinstance(block.get("text"), str):
                    raise _invalid(message="Text content must contain text")
                coverage.add(MessagesCoverage.TEXT)
            elif kind == "tool_result":
                coverage.update(self._validate_tool_result(block=block, role=role))
            elif kind == "tool_use":
                coverage.update(self._validate_tool_use(block=block, role=role))
            else:
                coverage.update(self._validate_thinking(block=block, role=role))
        return coverage

    def _validate_tool_result(self, *, block: dict[str, Any], role: str) -> set[MessagesCoverage]:
        if role != "user" or not self._capabilities.tool_use:
            raise _unsupported(message="Only the Claude client may return tool_result blocks")
        _string(value=block.get("tool_use_id"), where="tool_use_id")
        if "is_error" in block and type(block["is_error"]) is not bool:
            raise _invalid(message="tool_result.is_error must be a boolean")
        coverage = {MessagesCoverage.TOOL_RESULT}
        content = block.get("content", "")
        if isinstance(content, list):
            for part in content:
                item = _object(value=part, where="tool_result content")
                _fields(value=item, allowed=self._BLOCK_FIELDS["text"], where="tool_result content")
                if item.get("type") != "text" or not isinstance(item.get("text"), str):
                    raise _unsupported(message="Only text tool_result content is supported")
                if "cache_control" in item:
                    coverage.update(self._validate_cache_control(value=item["cache_control"]))
        elif not isinstance(content, str):
            raise _unsupported(message="Only text tool_result content is supported")
        return coverage

    def _validate_tool_use(self, *, block: dict[str, Any], role: str) -> set[MessagesCoverage]:
        if role != "assistant" or not self._capabilities.tool_use:
            raise _unsupported(message="Only Claude may emit client tool_use blocks")
        _string(value=block.get("id"), where="tool_use.id")
        _string(value=block.get("name"), where="tool_use.name")
        _object(value=block.get("input"), where="tool_use.input")
        return {MessagesCoverage.TOOL_USE}

    def _validate_thinking(self, *, block: dict[str, Any], role: str) -> set[MessagesCoverage]:
        if role != "assistant" or not self._capabilities.thinking:
            raise _unsupported(message="Model backend does not support signed thinking")
        if block["type"] == "thinking":
            if not isinstance(block.get("thinking"), str):
                raise _invalid(message="thinking text must be a string")
            _string(value=block.get("signature"), where="thinking.signature")
        else:
            _string(value=block.get("data"), where="redacted_thinking.data")
        return {MessagesCoverage.THINKING}

    def _validate_system(self, *, value: object) -> set[MessagesCoverage]:
        if isinstance(value, str):
            return {MessagesCoverage.TEXT}
        if not isinstance(value, list):
            raise _invalid(message="system must be text or an array of text blocks")
        coverage: set[MessagesCoverage] = set()
        for part in value:
            block = _object(value=part, where="system block")
            _fields(value=block, allowed=self._BLOCK_FIELDS["text"], where="system block")
            if block.get("type") != "text" or not isinstance(block.get("text"), str):
                raise _unsupported(message="Only system text blocks are supported")
            if "cache_control" in block:
                coverage.update(self._validate_cache_control(value=block["cache_control"]))
            coverage.add(MessagesCoverage.TEXT)
        return coverage

    def _validate_cache_control(self, *, value: object) -> set[MessagesCoverage]:
        if value is None:
            return set()
        if not self._capabilities.prompt_caching:
            raise _unsupported(message="Model backend does not support prompt caching")
        control = _object(value=value, where="cache_control")
        _fields(value=control, allowed=frozenset({"type", "ttl"}), where="cache_control")
        if control.get("type") != "ephemeral" or control.get("ttl", "5m") != "5m":
            raise _unsupported(message="Only five-minute ephemeral cache breakpoints are supported")
        return {MessagesCoverage.PROMPT_CACHING}

    def _validate_tools(self, *, value: object) -> tuple[set[str], set[MessagesCoverage]]:
        if not isinstance(value, list) or len(value) > 64:
            raise _invalid(message="tools must be an array of at most 64 client tools")
        if value and not self._capabilities.tool_use:
            raise _unsupported(message="Model backend does not support client tool use")
        names: set[str] = set()
        coverage: set[MessagesCoverage] = set()
        for tool in value:
            definition = _object(value=tool, where="tool definition")
            _fields(value=definition, allowed=self._TOOL_FIELDS, where="tool definition")
            if definition.get("type", "custom") != "custom":
                raise _unsupported(message="Model-hosted tools are not supported")
            name = _string(value=definition.get("name"), where="tool.name")
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", name) or name in names:
                raise _invalid(message="Client tool names must be unique, simple identifiers")
            schema = _object(value=definition.get("input_schema"), where="tool.input_schema")
            if schema.get("type") != "object":
                raise _unsupported(message="Only object-schema client tools are supported")
            if "description" in definition and not isinstance(definition["description"], str):
                raise _invalid(message="Tool description must be text")
            if "cache_control" in definition:
                coverage.update(self._validate_cache_control(value=definition["cache_control"]))
            names.add(name)
            coverage.add(MessagesCoverage.TOOL_DEFINITION)
        return names, coverage

    def _validate_options(self, *, body: dict[str, Any], names: set[str], max_tokens: int) -> None:
        if "thinking" in body:
            thinking = _object(value=body["thinking"], where="thinking")
            _fields(value=thinking, allowed=frozenset({"type", "budget_tokens", "display"}), where="thinking")
            kind = thinking.get("type")
            if kind not in ("enabled", "adaptive", "disabled"):
                raise _unsupported(message="Unsupported thinking mode")
            if kind != "disabled" and not self._capabilities.thinking:
                raise _unsupported(message="Model backend does not support signed thinking")
            if kind == "enabled" and (
                type(thinking.get("budget_tokens")) is not int or not 1024 <= thinking["budget_tokens"] < max_tokens
            ):
                raise _invalid(message="thinking.budget_tokens must be at least 1024 and below max_tokens")
            if kind != "enabled" and "budget_tokens" in thinking:
                raise _invalid(message="Only enabled thinking accepts budget_tokens")
            if "display" in thinking and thinking["display"] not in ("summarized", "omitted", None):
                raise _unsupported(message="Unsupported thinking display")
        if "output_config" in body:
            if not self._capabilities.effort:
                raise _unsupported(message="Model backend does not support output effort controls")
            output = _object(value=body["output_config"], where="output_config")
            _fields(value=output, allowed=frozenset({"effort"}), where="output_config")
            if output.get("effort") not in ("low", "medium", "high", "max"):
                raise _unsupported(message="Only documented output effort levels are supported")
        if "tool_choice" in body:
            choice = _object(value=body["tool_choice"], where="tool_choice")
            _fields(value=choice, allowed=frozenset({"type", "name", "disable_parallel_tool_use"}), where="tool_choice")
            kind = choice.get("type")
            if kind not in ("auto", "none", "any", "tool"):
                raise _unsupported(message="Unsupported tool_choice type")
            if kind in ("any", "tool") and not names:
                raise _invalid(message="tool_choice requires advertised client tools")
            if kind == "tool" and (not isinstance(choice.get("name"), str) or choice["name"] not in names):
                raise _invalid(message="tool_choice must name an advertised client tool")
            if kind != "tool" and "name" in choice:
                raise _invalid(message="Only a named tool_choice accepts a tool name")
            if "disable_parallel_tool_use" in choice and type(choice["disable_parallel_tool_use"]) is not bool:
                raise _invalid(message="disable_parallel_tool_use must be a boolean")
        if "metadata" in body:
            metadata = _object(value=body["metadata"], where="metadata")
            _fields(value=metadata, allowed=frozenset({"user_id"}), where="metadata")
            if "user_id" in metadata and not isinstance(metadata["user_id"], str):
                raise _invalid(message="metadata.user_id must be text")
        if "stop_sequences" in body:
            sequences = body["stop_sequences"]
            if (
                not isinstance(sequences, list)
                or len(sequences) > 64
                or any(not isinstance(item, str) or not item for item in sequences)
            ):
                raise _invalid(message="stop_sequences must be a bounded array of nonempty strings")
        if "service_tier" in body and body["service_tier"] not in ("auto", "standard_only"):
            raise _unsupported(message="Unsupported service tier")
        for name, maximum in (("temperature", 1.0), ("top_p", 1.0)):
            if name in body:
                value = body[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise _invalid(message=f"{name} must be a finite number")
                if not 0 <= value <= maximum:
                    raise _invalid(message=f"{name} is outside its documented range")
        if "top_k" in body and (type(body["top_k"]) is not int or body["top_k"] < 0):
            raise _invalid(message="top_k must be a nonnegative integer")

    def validate_provider_message(
        self, *, value: object, max_tokens: int, advertised_tools: frozenset[str], streaming_start: bool = False
    ) -> frozenset[MessagesCoverage]:
        """
        Validate the provider's actual message, model, tool identities, and usage.

        Args:
            value (object): Original provider Message object.
            max_tokens (int): Output-token reservation for this request.
            advertised_tools (frozenset[str]): Client-owned tool names.
            streaming_start (bool): Whether content and stop reason are still pending.

        Returns:
            frozenset[MessagesCoverage]: Supported provider features and completion status.

        Raises:
            GatewayError: If the backend emitted an invalid or unsupported Message.
        """
        message = _provider_object(value=value, where="message")
        if (
            message.get("type") != "message"
            or message.get("role") != "assistant"
            or message.get("model") != self._route.model
            or not isinstance(message.get("id"), str)
            or not message["id"]
        ):
            raise _invalid_provider(message="Model backend returned an unrouted Anthropic Message")
        content = message.get("content")
        if not isinstance(content, list) or (streaming_start and content):
            raise _invalid_provider(message="Model backend returned invalid message content")
        coverage: set[MessagesCoverage] = set()
        for entry in content:
            coverage.update(self.validate_provider_block(value=entry, advertised_tools=advertised_tools))
        usage = _provider_object(value=message.get("usage"), where="usage")
        _token_count(value=usage.get("input_tokens"), where="input_tokens", maximum=2**63 - 1)
        _token_count(value=usage.get("output_tokens"), where="output_tokens", maximum=max_tokens)
        reason = message.get("stop_reason")
        if streaming_start:
            if reason is not None:
                raise _invalid_provider(message="Streaming message_start must not contain a stop reason")
        elif reason not in self._STOP_REASONS:
            raise _invalid_provider(message="Model backend Message has no documented stop reason")
        elif reason == "tool_use" and MessagesCoverage.TOOL_USE not in coverage:
            raise _invalid_provider(message="Model backend stopped for tool use without a tool_use block")
        if not streaming_start:
            coverage.add(MessagesCoverage.COMPLETED)
        return frozenset(coverage)

    def validate_provider_block(
        self, *, value: object, advertised_tools: frozenset[str], starting: bool = False
    ) -> frozenset[MessagesCoverage]:
        """
        Check one model-origin content block, never invoking a tool.

        Args:
            value (object): Provider content block.
            advertised_tools (frozenset[str]): CLI-owned client tools from the request.
            starting (bool): Whether streaming content is still incomplete.

        Returns:
            frozenset[MessagesCoverage]: Features present in the block.

        Raises:
            GatewayError: If the backend returned an unsupported or undeclared block.
        """
        block = _provider_object(value=value, where="content block")
        kind = block.get("type")
        if kind == "text":
            if not isinstance(block.get("text"), str):
                raise _invalid_provider(message="Model backend text block has no text")
            return frozenset({MessagesCoverage.TEXT})
        if kind == "tool_use":
            if (
                not self._capabilities.tool_use
                or not isinstance(block.get("name"), str)
                or block["name"] not in advertised_tools
            ):
                raise _invalid_provider(message="Model backend called an undeclared client tool")
            if not isinstance(block.get("id"), str) or not block["id"] or not isinstance(block.get("input"), dict):
                raise _invalid_provider(message="Model backend returned invalid tool_use identity or input")
            return frozenset({MessagesCoverage.TOOL_USE})
        if kind in ("thinking", "redacted_thinking"):
            if not self._capabilities.thinking:
                raise _invalid_provider(message="Model backend returned unsupported thinking")
            field = "thinking" if kind == "thinking" else "data"
            if not isinstance(block.get(field), str):
                raise _invalid_provider(message="Model backend returned invalid thinking content")
            if kind == "thinking" and not starting and not isinstance(block.get("signature"), str):
                raise _invalid_provider(message="Model backend returned unsigned thinking")
            return frozenset({MessagesCoverage.THINKING})
        raise _invalid_provider(message="Model backend returned unsupported Anthropic content")

    def validate_provider_error(self, *, value: object) -> None:
        """
        Require a real Anthropic error envelope before forwarding provider bytes.

        Args:
            value (object): Strictly decoded original upstream error body.

        Raises:
            GatewayError: If the provider error envelope is malformed.
        """
        payload = _provider_object(value=value, where="error body")
        error = _provider_object(value=payload.get("error"), where="error")
        if (
            payload.get("type") != "error"
            or not isinstance(error.get("type"), str)
            or not error["type"]
            or not isinstance(error.get("message"), str)
            or not error["message"]
        ):
            raise _invalid_provider(message="Model backend returned an invalid Anthropic error")

    def validate_provider_headers(self, *, headers: tuple[tuple[str, str], ...], streaming: bool) -> None:
        """
        Accept only documented, credential-free Anthropic response metadata.

        Args:
            headers (tuple[tuple[str, str], ...]): Host-selected original provider headers.
            streaming (bool): Whether the response is an SSE stream.

        Raises:
            GatewayError: If content type or metadata is missing, ambiguous, or unsafe.
        """
        if not isinstance(headers, tuple) or len(headers) > 64:
            raise _invalid_provider(message="Model backend returned invalid response headers")
        content_types: list[str] = []
        seen: set[str] = set()
        for pair in headers:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise _invalid_provider(message="Model backend returned invalid response headers")
            name, value = pair
            if not isinstance(name, str) or not isinstance(value, str):
                raise _invalid_provider(message="Model backend returned invalid response headers")
            key = name.lower()
            if key in seen:
                raise _invalid_provider(message="Model backend returned duplicate response headers")
            seen.add(key)
            if key not in ("content-type", "retry-after", "x-should-retry") and not key.startswith(
                "anthropic-ratelimit-unified-"
            ):
                raise _invalid_provider(message="Model backend returned unsupported response headers")
            if len(value) > 512 or not value.isascii() or any(ord(char) < 32 or ord(char) > 126 for char in value):
                raise _invalid_provider(message="Model backend returned invalid response header values")
            if key == "content-type":
                content_types.append(value)
            elif key == "retry-after" and (not value.isascii() or not value.isdigit() or len(value) > 5):
                raise _invalid_provider(message="Model backend returned invalid retry-after")
            elif key == "x-should-retry" and value not in ("true", "false"):
                raise _invalid_provider(message="Model backend returned invalid x-should-retry")
        expected = "text/event-stream" if streaming else "application/json"
        if len(content_types) != 1 or content_types[0].split(";", maxsplit=1)[0].strip().lower() != expected:
            raise _invalid_provider(message="Model backend returned an unexpected content type")
