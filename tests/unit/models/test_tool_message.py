# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

import pytest

from pyrit.models import (
    ToolCallError,
    ToolCallRequest,
    ToolCallResult,
)


def test_tool_models_publish_stable_json_schemas() -> None:
    request_schema = ToolCallRequest.model_json_schema()
    result_schema = ToolCallResult.model_json_schema()

    assert request_schema["properties"]["type"]["const"] == "function_call"
    assert request_schema["required"] == ["call_id", "name", "arguments"]
    assert result_schema["properties"]["type"]["const"] == "function_call_output"
    assert result_schema["required"] == ["call_id", "output"]
    assert "error" in result_schema["properties"]
    assert "provenance" in request_schema["properties"]


def test_tool_call_request_normalizes_provider_wire_shapes() -> None:
    arguments = '{"city":"Seattle", "units":"f"}'
    chat = ToolCallRequest.from_openai_chat(
        call_id="call-1",
        name="weather",
        arguments=arguments,
        raw={
            "id": "call-1",
            "type": "function",
            "function": {"name": "weather", "arguments": arguments},
        },
    )
    responses = ToolCallRequest.from_openai_responses(
        call_id="call-1",
        name="weather",
        arguments=arguments,
        raw={
            "type": "function_call",
            "call_id": "call-1",
            "name": "weather",
            "arguments": arguments,
        },
    )

    assert chat.call_id == responses.call_id
    assert chat.name == responses.name
    assert chat.arguments == responses.arguments == arguments
    assert json.loads(chat.to_json())["type"] == "function_call"
    assert json.loads(responses.to_json())["type"] == "function_call"


def test_tool_call_request_parses_legacy_chat_and_responses_payloads() -> None:
    chat = ToolCallRequest.from_json(
        '{"type":"function","id":"chat-1","function":{"name":"lookup","arguments":"{\\"x\\":1}"}}'
    )
    responses = ToolCallRequest.from_json(
        '{"type":"function_call","call_id":"responses-1","name":"lookup","arguments":"{\\"x\\":1}"}'
    )

    assert chat.call_id == "chat-1"
    assert responses.call_id == "responses-1"
    assert chat.name == responses.name == "lookup"


@pytest.mark.parametrize(
    "payload",
    [
        '{"call_id":"call-1","name":"lookup","arguments":"{}"}',
        '{"type":"function_call_output","call_id":"call-1","name":"lookup","arguments":"{}"}',
    ],
)
def test_tool_call_request_rejects_missing_or_incompatible_type(payload: str) -> None:
    with pytest.raises(ValueError, match="type 'function_call'"):
        ToolCallRequest.from_json(payload)


def test_tool_call_request_serializes_to_both_provider_shapes() -> None:
    call = ToolCallRequest(call_id="call-1", name="lookup", arguments='{"x":1}')

    assert call.to_openai_chat_tool_call() == {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"x":1}'},
    }
    assert call.to_openai_responses_function_call() == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "lookup",
        "arguments": '{"x":1}',
    }


def test_tool_call_result_preserves_output_error_and_provider_shapes() -> None:
    result = ToolCallResult(
        call_id="call-1",
        output='{"error":"denied"}',
        error=ToolCallError(code="denied", message="Approval denied.", details={"approval_id": "a-1"}),
    )

    parsed = ToolCallResult.from_json(result.to_json())
    assert parsed == result
    assert parsed.error is not None
    assert parsed.error.details == {"approval_id": "a-1"}
    assert result.to_openai_chat_message() == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"error":"denied"}',
    }
    assert result.to_openai_responses_function_call_output() == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": '{"error":"denied"}',
    }


@pytest.mark.parametrize(
    "payload",
    [
        '{"call_id":"call-1","output":"done"}',
        '{"type":"function_call","call_id":"call-1","output":"done"}',
    ],
)
def test_tool_call_result_rejects_missing_or_incompatible_type(payload: str) -> None:
    with pytest.raises(ValueError, match="type 'function_call_output'"):
        ToolCallResult.from_json(payload)
