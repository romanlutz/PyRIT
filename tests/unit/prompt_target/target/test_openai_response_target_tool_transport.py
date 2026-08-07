# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import RateLimitError
from openai.types.responses import ResponseOutputText

from pyrit.models import Message, MessagePiece, TargetInvocation, ToolCallRequest, ToolCallResult
from pyrit.prompt_target import OpenAIResponsesRequestOptions, OpenAIResponseTarget

pytestmark = pytest.mark.usefixtures("patch_central_database")


def _function_call_section(*, call_id: str, name: str, arguments: str) -> MagicMock:
    section = MagicMock()
    section.type = "function_call"
    section.call_id = call_id
    section.name = name
    section.arguments = arguments
    return section


def _function_response(*, response_id: str, calls: list[MagicMock]) -> MagicMock:
    response = MagicMock()
    response.id = response_id
    response.status = "completed"
    response.error = None
    response.incomplete_details = None
    response.usage = None
    response.output = calls
    return response


def _text_response(*, response_id: str, text: str, usage: dict[str, object] | None = None) -> MagicMock:
    section = MagicMock()
    section.type = "message"
    section.content = [ResponseOutputText(annotations=[], text=text, type="output_text")]
    response = MagicMock()
    response.id = response_id
    response.status = "completed"
    response.error = None
    response.incomplete_details = None
    response.usage = usage
    response.output = [section]
    return response


def _request() -> Message:
    return Message(
        message_pieces=[
            MessagePiece(
                role="user",
                original_value="Use the tools",
                conversation_id="tool-transport",
            )
        ]
    )


async def test_single_generation_mode_returns_calls_without_executing_tools() -> None:
    execute = AsyncMock(return_value={"should": "not run"})
    with pytest.warns(DeprecationWarning):
        target = OpenAIResponseTarget(
            model_name="gpt-4o",
            endpoint="https://mock.azure.com",
            api_key="key",
            custom_functions={"lookup": execute},
        )
    response = _function_response(
        response_id="resp-one",
        calls=[_function_call_section(call_id="call-1", name="lookup", arguments='{"x":1}')],
    )

    with patch.object(target._client.responses, "create", new=AsyncMock(return_value=response)) as create:
        messages = await target.send_prompt_async(
            message=_request(),
            request_options=OpenAIResponsesRequestOptions(tool_execution_mode="single_generation"),
        )

    assert create.await_count == 1
    execute.assert_not_awaited()
    assert len(messages) == 1
    call = ToolCallRequest.from_json(messages[0].message_pieces[0].original_value)
    assert (call.call_id, call.name, call.arguments) == ("call-1", "lookup", '{"x":1}')


async def test_legacy_mode_executes_every_call_in_response_order() -> None:
    execution_order: list[str] = []

    async def first(arguments: dict[str, object]) -> dict[str, object]:
        execution_order.append("first")
        return {"value": arguments["value"]}

    async def second(arguments: dict[str, object]) -> dict[str, object]:
        execution_order.append("second")
        return {"value": arguments["value"]}

    with pytest.warns(DeprecationWarning):
        target = OpenAIResponseTarget(
            model_name="gpt-4o",
            endpoint="https://mock.azure.com",
            api_key="key",
            custom_functions={"first": first, "second": second},
        )
    responses = [
        _function_response(
            response_id="resp-calls",
            calls=[
                _function_call_section(call_id="call-1", name="first", arguments='{"value":1}'),
                _function_call_section(call_id="call-2", name="second", arguments='{"value":2}'),
            ],
        ),
        _text_response(response_id="resp-final", text="done"),
    ]

    with patch.object(target._client.responses, "create", new=AsyncMock(side_effect=responses)) as create:
        messages = await target.send_prompt_async(message=_request())

    assert execution_order == ["first", "second"]
    assert len(messages) == 3
    assert [ToolCallRequest.from_json(piece.original_value).call_id for piece in messages[0].message_pieces] == [
        "call-1",
        "call-2",
    ]
    assert [ToolCallResult.from_json(piece.original_value).call_id for piece in messages[1].message_pieces] == [
        "call-1",
        "call-2",
    ]
    assert messages[2].get_value() == "done"
    second_generation_input = create.await_args_list[1].kwargs["input"]
    assert [item["call_id"] for item in second_generation_input[1:]] == [
        "call-1",
        "call-2",
        "call-1",
        "call-2",
    ]


async def test_retrying_later_generation_does_not_replay_tool_side_effect() -> None:
    side_effect_count = 0

    async def side_effect_tool(arguments: dict[str, object]) -> dict[str, object]:
        nonlocal side_effect_count
        side_effect_count += 1
        return {"count": side_effect_count}

    with pytest.warns(DeprecationWarning):
        target = OpenAIResponseTarget(
            model_name="gpt-4o",
            endpoint="https://mock.azure.com",
            api_key="key",
            custom_functions={"side_effect": side_effect_tool},
        )
    rate_limit = RateLimitError("limited", response=MagicMock(status_code=429), body=None)
    provider_results = [
        _function_response(
            response_id="resp-call",
            calls=[_function_call_section(call_id="call-1", name="side_effect", arguments="{}")],
        ),
        rate_limit,
        _text_response(response_id="resp-final", text="done"),
    ]

    with patch.object(target._client.responses, "create", new=AsyncMock(side_effect=provider_results)) as create:
        messages = await target.send_prompt_async(message=_request())

    assert create.await_count == 3
    assert side_effect_count == 1
    assert messages[-1].get_value() == "done"


async def test_response_identity_stop_reason_and_usage_are_recorded() -> None:
    target = OpenAIResponseTarget(
        model_name="gpt-4o",
        endpoint="https://mock.azure.com",
        api_key="key",
    )
    request = _request()
    response = _text_response(
        response_id="resp-metadata",
        text="answer",
        usage={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
    )

    with patch.object(target._client.responses, "create", new=AsyncMock(return_value=response)):
        messages = await target.send_prompt_async(message=request)

    invocation = TargetInvocation.from_metadata(metadata=request.message_pieces[0].prompt_metadata)
    assert invocation is not None
    assert len(invocation.responses) == 1
    assert invocation.responses[0].provider_response_id == "resp-metadata"
    assert invocation.responses[0].stop_reason == "completed"
    assert invocation.responses[0].provider_stop_reason == "completed"
    assert invocation.responses[0].usage is not None
    assert invocation.responses[0].usage.total_tokens == 7
    assert messages[0].message_pieces[0].prompt_metadata["token_usage_total_tokens"] == 7


async def test_truncated_tool_call_records_length_stop_reason() -> None:
    target = OpenAIResponseTarget(
        model_name="gpt-4o",
        endpoint="https://mock.azure.com",
        api_key="key",
    )
    request = _request()
    response = _function_response(
        response_id="resp-truncated",
        calls=[_function_call_section(call_id="call-partial", name="lookup", arguments='{"x":')],
    )
    response.status = "incomplete"
    response.incomplete_details = MagicMock(reason="max_output_tokens")

    with patch.object(target._client.responses, "create", new=AsyncMock(return_value=response)):
        messages = await target.send_prompt_async(message=request)

    invocation = TargetInvocation.from_metadata(metadata=request.message_pieces[0].prompt_metadata)
    assert invocation is not None
    assert invocation.responses[0].stop_reason == "length"
    assert messages[0].message_pieces[0].is_truncated
    assert all(piece.original_value_data_type != "function_call" for piece in messages[0].message_pieces)
