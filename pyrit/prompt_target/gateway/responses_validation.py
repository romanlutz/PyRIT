# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# DOC501 mistakes exception-factory calls for separate exception types; all raise GatewayError.
# ruff: noqa: DOC501

"""Fail-closed validation of the text/tool subset of the Responses wire API."""

import json
import math
from dataclasses import dataclass
from typing import Any, NoReturn

from pyrit.prompt_target.gateway.responses_contract import (
    BackendCapabilities,
    GatewayCoverage,
    GatewayError,
    GatewayLimits,
    GatewayRoute,
    ModelRequest,
)


def _invalid(*, message: str, param: str | None = None) -> GatewayError:
    return GatewayError(status_code=400, code="invalid_request", message=message, param=param)


def _unsupported(*, message: str, param: str | None = None) -> GatewayError:
    return GatewayError(status_code=501, code="unsupported_feature", message=message, param=param)


def _invalid_backend(*, message: str) -> GatewayError:
    return GatewayError(status_code=502, code="invalid_backend_response", message=message)


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError("Non-finite JSON numbers are not supported")


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON numbers are not supported")
    return number


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON fields are not supported")
        result[key] = value
    return result


def strict_json_loads(*, value: bytes | str) -> Any:
    """
    Reject duplicate fields and non-finite numbers before validating wire shapes.

    Args:
        value (bytes | str): Original JSON bytes or SSE event data.

    Returns:
        Any: Parsed JSON value with standard finite numbers and unique object keys.
    """
    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
        object_pairs_hook=_unique_json_object,
    )


def _check_fields(*, value: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise _unsupported(message=f"Unsupported {where} field(s): {', '.join(sorted(extra))}", param=where)


def _string(*, value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(message=f"{where} must be a nonempty string", param=where)
    return value


def _object(*, value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _invalid(message=f"{where} must be a JSON object", param=where)
    return value


def _backend_object(*, value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _invalid_backend(message=f"Model backend {where} must be a JSON object")
    return value


@dataclass(frozen=True)
class ValidatedRequest:
    """A checked request and the original wire's observed features."""

    body: dict[str, Any]
    streaming: bool
    output_token_limit: int
    coverage: frozenset[GatewayCoverage]


class ResponsesValidator:
    """Validate only the Responses fields a sandboxed Codex run may send."""

    _REQUEST_FIELDS = {
        "model",
        "input",
        "instructions",
        "stream",
        "store",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "include",
        "text",
        "max_output_tokens",
        "temperature",
        "top_p",
    }
    _SSE_EVENTS = {
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.refusal.delta",
        "response.completed",
        "response.incomplete",
        "response.failed",
    }

    def __init__(self, *, route: GatewayRoute, limits: GatewayLimits, capabilities: BackendCapabilities) -> None:
        """Bind validation to exactly one model, run, and backend capability set."""
        self._route = route
        self._limits = limits
        self._capabilities = capabilities

    def validate_request(self, *, body: dict[str, Any]) -> ValidatedRequest:
        """
        Check the allowed wire subset without rewriting guest tool history.

        Args:
            body (dict[str, Any]): Decoded original JSON request.

        Returns:
            ValidatedRequest: Routed request, bounded token limit, and coverage.

        Raises:
            GatewayError: If the request uses unsupported or invalid features.
        """
        _check_fields(value=body, allowed=self._REQUEST_FIELDS, where="request")
        if body.get("model") != self._route.model:
            raise _invalid(message="This model is not routed to this run", param="model")
        if "input" not in body:
            raise _invalid(message="input is required", param="input")
        if "instructions" in body and not isinstance(body["instructions"], str):
            raise _invalid(message="instructions must be text", param="instructions")
        if "stream" in body and type(body["stream"]) is not bool:
            raise _invalid(message="stream must be a boolean", param="stream")
        if body.get("store", False) is not False:
            raise _unsupported(message="Stored/server-side Responses history is not supported", param="store")

        streaming = body.get("stream", False)
        if streaming and not self._capabilities.streaming:
            raise _unsupported(message="Model backend does not support ordered Responses SSE streaming", param="stream")

        coverage = self._validate_input(value=body["input"])
        coverage.update(self._validate_tools(value=body.get("tools", [])))
        self._validate_options(body=body)
        if "reasoning" in body or "include" in body:
            coverage.add(GatewayCoverage.REASONING)
        if streaming:
            coverage.add(GatewayCoverage.STREAMING)
        token_limit = self._output_token_limit(body=body)
        return ValidatedRequest(
            body={**body, "max_output_tokens": token_limit},
            streaming=streaming,
            output_token_limit=token_limit,
            coverage=frozenset(coverage),
        )

    def _validate_input(self, *, value: object) -> set[GatewayCoverage]:
        if isinstance(value, str) and value:
            return set()
        if not isinstance(value, list) or not value:
            raise _invalid(message="input must be nonempty text or an array of Responses input items", param="input")
        coverage: set[GatewayCoverage] = set()
        for item in value:
            entry = _object(value=item, where="input item")
            item_type = entry.get("type", "message")
            if not isinstance(item_type, str):
                raise _invalid(message="input item type must be text", param="input")
            if item_type == "message":
                self._validate_message(item=entry)
            elif item_type in {"function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output"}:
                coverage.add(self._validate_tool_history(item=entry, item_type=item_type))
            elif item_type == "reasoning":
                if not self._capabilities.reasoning:
                    raise _unsupported(message="Model backend cannot replay encrypted reasoning", param="input")
                _check_fields(
                    value=entry,
                    allowed={"type", "id", "summary", "encrypted_content", "status"},
                    where="reasoning input",
                )
                _string(value=entry.get("id"), where="reasoning id")
                _string(value=entry.get("encrypted_content"), where="encrypted reasoning")
                summary = entry.get("summary", [])
                if not isinstance(summary, list) or any(
                    not isinstance(part, dict)
                    or set(part) != {"type", "text"}
                    or part.get("type") != "summary_text"
                    or not isinstance(part.get("text"), str)
                    for part in summary
                ):
                    raise _invalid(message="reasoning summary must contain only summary_text items", param="input")
                coverage.add(GatewayCoverage.REASONING)
            else:
                raise _unsupported(message=f"Responses input type '{item_type}' is not supported", param="input")
        return coverage

    def _validate_message(self, *, item: dict[str, Any]) -> None:
        _check_fields(value=item, allowed={"type", "role", "content", "id", "status"}, where="message input")
        if item.get("role") not in ("system", "developer", "user", "assistant"):
            raise _invalid(message="message role must be system, developer, user, or assistant", param="input")
        self._validate_item_identity(item=item)
        content = item.get("content")
        if isinstance(content, str):
            return
        if not isinstance(content, list) or not content:
            raise _invalid(message="message content must be text or a nonempty list", param="input")
        for part in content:
            section = _object(value=part, where="message content")
            part_type = section.get("type")
            if not isinstance(part_type, str):
                raise _invalid(message="message content type must be text", param="input")
            if part_type in {"input_text", "output_text"}:
                _check_fields(value=section, allowed={"type", "text", "annotations"}, where="text content")
                if not isinstance(section.get("text"), str):
                    raise _invalid(message="text content must contain a string", param="input")
                if "annotations" in section and section["annotations"] != []:
                    raise _unsupported(message="Content annotations are not supported", param="input")
            elif part_type == "refusal":
                _check_fields(value=section, allowed={"type", "refusal"}, where="refusal content")
                _string(value=section.get("refusal"), where="refusal")
            else:
                raise _unsupported(message=f"Responses content type '{part_type}' is not supported", param="input")

    def _validate_item_identity(self, *, item: dict[str, Any]) -> None:
        if "id" in item:
            _string(value=item["id"], where="item id")
        if "status" in item and item["status"] not in ("completed", "in_progress", "incomplete"):
            raise _invalid(message="Unsupported input item status", param="input")

    def _validate_tool_history(self, *, item: dict[str, Any], item_type: str) -> GatewayCoverage:
        custom = item_type.startswith("custom")
        if not (self._capabilities.custom_tools if custom else self._capabilities.function_tools):
            raise _unsupported(message=f"Model backend does not support {item_type} history", param="input")
        is_result = item_type.endswith("_output")
        allowed = (
            {"type", "call_id", "output", "id", "status"}
            if is_result
            else {
                "type",
                "call_id",
                "name",
                "input" if custom else "arguments",
                "id",
                "status",
            }
        )
        _check_fields(value=item, allowed=allowed, where=item_type)
        self._validate_item_identity(item=item)
        _string(value=item.get("call_id"), where="call_id")
        if is_result:
            if not isinstance(item.get("output"), str):
                raise _invalid(message="tool output must be a string; the CLI owns tool execution", param="input")
            return GatewayCoverage.CUSTOM_RESULT if custom else GatewayCoverage.FUNCTION_RESULT
        _string(value=item.get("name"), where="tool name")
        field_name = "input" if custom else "arguments"
        if not isinstance(item.get(field_name), str):
            raise _invalid(message=f"{field_name} must be a string", param="input")
        return GatewayCoverage.CUSTOM_CALL if custom else GatewayCoverage.FUNCTION_CALL

    def _validate_tools(self, *, value: object) -> set[GatewayCoverage]:
        if not isinstance(value, list):
            raise _invalid(message="tools must be an array", param="tools")
        if len(value) > 64:
            raise _invalid(message="At most 64 tools may be advertised", param="tools")
        coverage: set[GatewayCoverage] = set()
        names: set[str] = set()
        for tool in value:
            definition = _object(value=tool, where="tool definition")
            tool_type = definition.get("type")
            if not isinstance(tool_type, str):
                raise _invalid(message="tool type must be text", param="tools")
            if tool_type not in {"function", "custom"}:
                raise _unsupported(message=f"Model-hosted tool type '{tool_type}' is not supported", param="tools")
            custom = tool_type == "custom"
            if not (self._capabilities.custom_tools if custom else self._capabilities.function_tools):
                raise _unsupported(message=f"Model backend does not support {tool_type} tool calls", param="tools")
            self._validate_tool_definition(definition=definition, custom=custom)
            name = _string(value=definition.get("name"), where="tool name")
            if name in names:
                raise _invalid(message="Tool names must be unique", param="tools")
            names.add(name)
            coverage.add(GatewayCoverage.CUSTOM_TOOL if custom else GatewayCoverage.FUNCTION_TOOL)
        return coverage

    def _validate_tool_definition(self, *, definition: dict[str, Any], custom: bool) -> None:
        fields = (
            {"type", "name", "description", "format"}
            if custom
            else {
                "type",
                "name",
                "description",
                "parameters",
                "strict",
            }
        )
        _check_fields(value=definition, allowed=fields, where="tool definition")
        if "description" in definition and not isinstance(definition["description"], str):
            raise _invalid(message="tool description must be text", param="tools")
        if custom:
            format_spec = _object(value=definition.get("format"), where="custom tool format")
            if format_spec != {"type": "text"}:
                raise _unsupported(message="Only text-format custom tools are supported", param="tools")
        else:
            _object(value=definition.get("parameters"), where="function parameters")
            if "strict" in definition and type(definition["strict"]) is not bool:
                raise _invalid(message="function strict must be a boolean", param="tools")

    def _validate_options(self, *, body: dict[str, Any]) -> None:
        if "reasoning" in body:
            if not self._capabilities.reasoning:
                raise _unsupported(message="Model backend does not support reasoning", param="reasoning")
            reasoning = _object(value=body["reasoning"], where="reasoning")
            _check_fields(value=reasoning, allowed={"effort", "summary"}, where="reasoning")
            if "effort" in reasoning and reasoning["effort"] not in (
                "none",
                "minimal",
                "low",
                "medium",
                "high",
                "xhigh",
            ):
                raise _invalid(message="Unsupported reasoning effort", param="reasoning")
            if "summary" in reasoning and reasoning["summary"] not in ("auto", "concise", "detailed"):
                raise _invalid(message="Unsupported reasoning summary", param="reasoning")
        if "include" in body and (
            not self._capabilities.reasoning or body["include"] != ["reasoning.encrypted_content"]
        ):
            raise _unsupported(message="Only reasoning.encrypted_content inclusion is supported", param="include")
        if "text" in body:
            text = _object(value=body["text"], where="text")
            _check_fields(value=text, allowed={"format", "verbosity"}, where="text")
            if "format" in text and text["format"] != {"type": "text"}:
                raise _unsupported(message="Only plain-text Responses formatting is supported", param="text")
            if "verbosity" in text and text["verbosity"] not in ("low", "medium", "high"):
                raise _invalid(message="Unsupported text verbosity", param="text")
        if "parallel_tool_calls" in body and type(body["parallel_tool_calls"]) is not bool:
            raise _invalid(message="parallel_tool_calls must be a boolean", param="parallel_tool_calls")
        self._validate_tool_choice(body=body)
        for name, maximum in (("temperature", 2.0), ("top_p", 1.0)):
            if name not in body:
                continue
            value = body[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise _invalid(message=f"{name} must be a finite number", param=name)
            if not 0 <= value <= maximum:
                raise _invalid(message=f"{name} must be between 0 and {maximum}", param=name)

    def _validate_tool_choice(self, *, body: dict[str, Any]) -> None:
        choice = body.get("tool_choice", "auto")
        if isinstance(choice, str):
            if choice not in {"auto", "none", "required"}:
                raise _unsupported(message="Unsupported tool_choice", param="tool_choice")
            return
        selected = _object(value=choice, where="tool_choice")
        _check_fields(value=selected, allowed={"type", "name"}, where="tool_choice")
        kind = selected.get("type")
        if kind not in ("function", "custom"):
            raise _unsupported(message="Only function/custom tool_choice is supported", param="tool_choice")
        if not any(tool["type"] == kind and tool["name"] == selected.get("name") for tool in body.get("tools", [])):
            raise _invalid(message="tool_choice must name an advertised tool", param="tool_choice")

    def _output_token_limit(self, *, body: dict[str, Any]) -> int:
        tokens = body.get("max_output_tokens", self._limits.max_output_tokens_per_request)
        if type(tokens) is not int or tokens <= 0:
            raise _invalid(message="max_output_tokens must be a positive integer", param="max_output_tokens")
        if tokens > self._limits.max_output_tokens_per_request:
            raise GatewayError(
                status_code=429,
                code="output_token_limit",
                message="Requested max_output_tokens exceeds this run's per-request budget",
                param="max_output_tokens",
            )
        return tokens

    def validate_response(self, *, frame: bytes, request: ModelRequest) -> frozenset[GatewayCoverage]:
        """
        Check a raw non-streaming response without changing its bytes.

        Args:
            frame (bytes): Original model backend response.
            request (ModelRequest): Validated originating request.

        Returns:
            frozenset[GatewayCoverage]: Model output features and terminal state.

        Raises:
            GatewayError: If model output cannot be safely forwarded.
        """
        try:
            data = strict_json_loads(value=frame)
        except (TypeError, ValueError, RecursionError) as exc:
            raise _invalid_backend(message="Model backend returned invalid Responses JSON") from exc
        if not isinstance(data, dict):
            raise _invalid_backend(message="Model backend response must be a JSON object")
        return self.validate_response_object(response=data, request=request, is_terminal=True)

    def validate_response_object(
        self, *, response: dict[str, Any], request: ModelRequest, is_terminal: bool
    ) -> frozenset[GatewayCoverage]:
        """
        Check response identity, tool output, and reported token usage.

        Args:
            response (dict[str, Any]): Provider response object.
            request (ModelRequest): Validated originating request.
            is_terminal (bool): Whether usage and a terminal status are required.

        Returns:
            frozenset[GatewayCoverage]: Verified output features and terminal state.

        Raises:
            GatewayError: If the response violates the backend contract.
        """
        if response.get("object") != "response" or not isinstance(response.get("id"), str) or not response["id"]:
            raise _invalid_backend(message="Model backend returned a response without a Responses id")
        if response.get("model") != request.body["model"]:
            raise _invalid_backend(message="Model backend response model does not match this run")
        status = response.get("status")
        if status not in ("completed", "incomplete", "failed", "in_progress"):
            raise _invalid_backend(message="Model backend returned an unsupported response status")
        if response.get("error") is not None and status != "failed":
            raise _invalid_backend(message="Model backend response reported an error without a failed status")
        if is_terminal and status == "in_progress":
            raise _invalid_backend(message="Model backend response never completed")
        if not is_terminal and status != "in_progress":
            raise _invalid_backend(message="Streaming response.created has an invalid status")
        coverage: set[GatewayCoverage] = set()
        output = response.get("output", [])
        if not isinstance(output, list):
            raise _invalid_backend(message="Model backend response output must be an array")
        for item in output:
            coverage.update(self.validate_output_item(item=item, request=request, finished=True))
        if is_terminal and status != "failed":
            self._validate_usage(response=response, token_limit=request.output_token_limit)
        if status == "completed":
            coverage.add(GatewayCoverage.COMPLETED)
        elif status == "incomplete":
            coverage.add(GatewayCoverage.INCOMPLETE)
        elif status == "failed":
            coverage.add(GatewayCoverage.FAILED)
        return frozenset(coverage)

    def validate_output_item(
        self, *, item: object, request: ModelRequest, finished: bool
    ) -> frozenset[GatewayCoverage]:
        """
        Reject model-hosted tools and preserve declared CLI tool call identities.

        Args:
            item (object): Output item supplied by the model-only backend.
            request (ModelRequest): Validated originating request with declared tools.
            finished (bool): Whether the item contains its final content.

        Returns:
            frozenset[GatewayCoverage]: Output features found in this item.

        Raises:
            GatewayError: If the backend returned malformed or undeclared tool output.
        """
        output = _backend_object(value=item, where="output item")
        kind = output.get("type")
        if kind == "message":
            if finished and (
                output.get("role") != "assistant"
                or not isinstance(output.get("content"), list)
                or any(
                    not isinstance(content, dict)
                    or content.get("type") not in ("output_text", "refusal")
                    or not isinstance(content.get("text" if content.get("type") == "output_text" else "refusal"), str)
                    for content in output["content"]
                )
            ):
                raise _invalid_backend(message="Model backend returned unsupported message content")
            return frozenset[GatewayCoverage]()
        if kind == "reasoning":
            if not self._capabilities.reasoning:
                raise _invalid_backend(message="Model backend emitted unsupported reasoning output")
            return frozenset({GatewayCoverage.REASONING})
        if kind not in ("function_call", "custom_tool_call"):
            raise _invalid_backend(message=f"Model backend emitted unsupported output type '{kind}'")
        custom = kind == "custom_tool_call"
        if not (self._capabilities.custom_tools if custom else self._capabilities.function_tools):
            raise _invalid_backend(message="Model backend emitted an unadvertised tool capability")
        if finished:
            tool_type = "custom" if custom else "function"
            tools = request.body.get("tools", [])
            if not any(tool["type"] == tool_type and tool["name"] == output.get("name") for tool in tools):
                raise _invalid_backend(message="Model backend called a tool not advertised by this request")
            value_field = "input" if custom else "arguments"
            if not isinstance(output.get("call_id"), str) or not output["call_id"]:
                raise _invalid_backend(message="Model backend tool call has no call_id")
            if not isinstance(output.get(value_field), str):
                raise _invalid_backend(message="Model backend tool call has no string arguments")
        return frozenset({GatewayCoverage.CUSTOM_CALL if custom else GatewayCoverage.FUNCTION_CALL})

    def _validate_usage(self, *, response: dict[str, Any], token_limit: int) -> None:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise _invalid_backend(message="Model backend must report usage to enforce the token budget")
        tokens = usage.get("output_tokens")
        if type(tokens) is not int or tokens < 0 or tokens > token_limit:
            raise _invalid_backend(message="Model backend exceeded or omitted the reserved output-token limit")


class ResponsesStreamValidator:
    """Check each complete SSE frame without buffering or reordering it."""

    def __init__(self, *, validator: ResponsesValidator, request: ModelRequest, max_bytes: int) -> None:
        """Track a single model response's byte budget and event ordering."""
        self._validator = validator
        self._request = request
        self._max_bytes = max_bytes
        self._bytes_seen = 0
        self._events_seen = 0
        self._response_id: str | None = None
        self._last_sequence: int | None = None
        self.terminal = False
        self.done = False

    def accept(self, *, frame: bytes) -> frozenset[GatewayCoverage]:
        """
        Validate and account for exactly one original SSE frame.

        Args:
            frame (bytes): Complete original SSE event or [DONE] frame.

        Returns:
            frozenset[GatewayCoverage]: Features witnessed on this frame.

        Raises:
            GatewayError: If the backend exceeded the byte limit or broke SSE semantics.
        """
        if not isinstance(frame, bytes):
            raise _invalid_backend(message="Model backend SSE frames must be bytes")
        self._bytes_seen += len(frame)
        if self._bytes_seen > self._max_bytes:
            raise GatewayError(status_code=502, code="response_too_large", message="Model response exceeded byte limit")
        event, data = self._parse_frame(frame=frame)
        if event == "[DONE]":
            if not self.terminal or self.done:
                raise _invalid_backend(message="Responses SSE [DONE] arrived before a terminal response")
            self.done = True
            return frozenset({GatewayCoverage.STREAMING})
        if self.terminal:
            raise _invalid_backend(message="Responses SSE emitted an event after the terminal response")
        if event not in ResponsesValidator._SSE_EVENTS:
            raise _invalid_backend(message=f"Unsupported Responses SSE event '{event}'")
        if self._events_seen == 0 and event != "response.created":
            raise _invalid_backend(message="Responses SSE must begin with response.created")
        self._events_seen += 1
        self._check_sequence(data=data)
        coverage = set(self._check_event(event=event, data=data))
        coverage.add(GatewayCoverage.STREAMING)
        return frozenset(coverage)

    def _parse_frame(self, *, frame: bytes) -> tuple[str, dict[str, Any]]:
        try:
            text = frame.decode("utf-8").replace("\r\n", "\n")
        except UnicodeDecodeError as exc:
            raise _invalid_backend(message="Model backend SSE frame is not UTF-8") from exc
        if not text.endswith("\n\n"):
            raise _invalid_backend(message="Model backend must yield complete SSE frames")
        lines = text[:-2].split("\n")
        if lines == ["data: [DONE]"]:
            return "[DONE]", {}
        if len(lines) != 2 or not lines[0].startswith("event: ") or not lines[1].startswith("data: "):
            raise _invalid_backend(message="Model backend yielded an unsupported SSE frame")
        event = lines[0][7:]
        try:
            data = strict_json_loads(value=lines[1][6:])
        except (ValueError, RecursionError) as exc:
            raise _invalid_backend(message="Model backend yielded invalid SSE JSON") from exc
        if not isinstance(data, dict) or data.get("type") != event:
            raise _invalid_backend(message="Responses SSE event name and data.type must match")
        return event, data

    def _check_sequence(self, *, data: dict[str, Any]) -> None:
        sequence = data.get("sequence_number")
        if sequence is None:
            return
        if (
            type(sequence) is not int
            or sequence < 0
            or (self._last_sequence is not None and sequence <= self._last_sequence)
        ):
            raise _invalid_backend(message="Responses SSE sequence_number went backwards")
        self._last_sequence = sequence

    def _check_event(self, *, event: str, data: dict[str, Any]) -> frozenset[GatewayCoverage]:
        if event in {"response.created", "response.in_progress"}:
            response = _backend_object(value=data.get("response"), where="stream response")
            coverage = self._validator.validate_response_object(
                response=response, request=self._request, is_terminal=False
            )
            if self._response_id is None:
                self._response_id = response["id"]
            elif response["id"] != self._response_id:
                raise _invalid_backend(message="Responses SSE changed response id")
            return coverage
        if event in {"response.completed", "response.incomplete", "response.failed"}:
            response = _backend_object(value=data.get("response"), where="terminal response")
            if response.get("id") != self._response_id or response.get("status") != event.removeprefix("response."):
                raise _invalid_backend(message="Responses SSE terminal status or id does not match")
            coverage = self._validator.validate_response_object(
                response=response, request=self._request, is_terminal=True
            )
            self.terminal = True
            return coverage
        if event in {"response.output_item.added", "response.output_item.done"}:
            return self._validator.validate_output_item(
                item=data.get("item"), request=self._request, finished=event.endswith(".done")
            )
        if event in {"response.content_part.added", "response.content_part.done"}:
            part = _backend_object(value=data.get("part"), where="response content part")
            if part.get("type") not in ("output_text", "refusal"):
                raise _invalid_backend(message="Model backend emitted unsupported response content")
        if event.startswith("response.reasoning_summary"):
            if not self._validator._capabilities.reasoning:
                raise _invalid_backend(message="Model backend emitted unsupported reasoning event")
            return frozenset({GatewayCoverage.REASONING})
        if event.startswith("response.function_call_arguments") and not self._validator._capabilities.function_tools:
            raise _invalid_backend(message="Model backend emitted unsupported function-call event")
        if event.startswith("response.custom_tool_call_input") and not self._validator._capabilities.custom_tools:
            raise _invalid_backend(message="Model backend emitted unsupported custom-tool event")
        return frozenset[GatewayCoverage]()
