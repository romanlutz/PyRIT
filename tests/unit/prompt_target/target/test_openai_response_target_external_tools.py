# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from pyrit.memory import CentralMemory
from pyrit.models import MessagePiece
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import OpenAIResponseTarget
from tests.unit.mocks import openai_response_json_dict


def _tool_response() -> dict[str, Any]:
    response = openai_response_json_dict()
    response["output"] = [
        {
            "type": "function_call",
            "name": "fixture",
            "arguments": '{"value":"inert"}',
            "call_id": "call-offline-1",
            "id": "function-offline-1",
            "status": "completed",
        }
    ]
    return response


@pytest.mark.usefixtures("patch_central_database")
async def test_external_tools_return_one_response_without_callback_async() -> None:
    requests = []
    callback = AsyncMock()

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_tool_response() if len(requests) == 1 else openai_response_json_dict())

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        target = OpenAIResponseTarget(
            endpoint="https://fixture.invalid/v1",
            api_key="offline",
            model_name="offline-fixture",
            custom_functions={"fixture": callback},
            auto_execute_tools=False,
            httpx_client_kwargs={"http_client": client, "max_retries": 0},
        )
        normalizer = PromptNormalizer()
        response = await normalizer.send_prompt_async(
            message=MessagePiece(role="user", original_value="OFFLINE/SIMULATED").to_message(),
            target=target,
            conversation_id="offline-external-tools",
        )
        assert json.loads(response.get_value())["call_id"] == "call-offline-1"
        assert len(requests) == 1
        callback.assert_not_awaited()
        assert target.auto_execute_tools is False
        assert target.get_identifier().params["auto_execute_tools"] is False
        feedback = MessagePiece(
            role="tool",
            original_value_data_type="function_call_output",
            original_value=json.dumps(
                {"type": "function_call_output", "call_id": "call-offline-1", "output": "exact\nfeedback"}
            ),
        ).to_message()
        await normalizer.send_prompt_async(message=feedback, target=target, conversation_id="offline-external-tools")
    assert len(requests) == 2
    inputs = requests[1]["input"]
    assert [item["type"] for item in inputs if "type" in item] == ["function_call", "function_call_output"]
    assert inputs[-1]["output"] == "exact\nfeedback"
    messages = CentralMemory.get_memory_instance().get_conversation_messages(conversation_id="offline-external-tools")
    assert len(messages) == 4


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("feedback", ["exact\nfeedback", {"value": "unchanged", "number": 3}])
async def test_default_loop_preserves_string_and_dictionary_feedback_async(feedback: str | dict[str, Any]) -> None:
    requests = []
    callback = AsyncMock(return_value=feedback)

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_tool_response() if len(requests) == 1 else openai_response_json_dict())

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        target = OpenAIResponseTarget(
            endpoint="https://fixture.invalid/v1",
            api_key="offline",
            model_name="offline-fixture",
            custom_functions={"fixture": callback},
            httpx_client_kwargs={"http_client": client, "max_retries": 0},
        )
        await PromptNormalizer().send_prompt_async(
            message=MessagePiece(role="user", original_value="OFFLINE/SIMULATED").to_message(), target=target
        )
    callback.assert_awaited_once()
    assert len(requests) == 2
    assert target.auto_execute_tools is True
    assert "auto_execute_tools" not in target.get_identifier().params
    expected = feedback if isinstance(feedback, str) else json.dumps(feedback, separators=(",", ":"))
    assert requests[1]["input"][-1]["output"] == expected


@pytest.mark.usefixtures("patch_central_database")
async def test_continuation_uses_retained_history_without_new_input_or_conversion_async() -> None:
    requests = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=openai_response_json_dict() if len(requests) == 1 else _tool_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        target = OpenAIResponseTarget(
            endpoint="https://fixture.invalid/v1",
            api_key="offline",
            model_name="offline-fixture",
            auto_execute_tools=False,
            httpx_client_kwargs={"http_client": client, "max_retries": 0},
        )
        normalizer = PromptNormalizer()
        await normalizer.send_prompt_async(
            message=MessagePiece(role="user", original_value="OFFLINE/SIMULATED").to_message(),
            target=target,
            conversation_id="offline-continuation",
        )
        before = list(normalizer.memory.get_conversation_messages(conversation_id="offline-continuation"))
        with patch.object(normalizer, "convert_values_async", new_callable=AsyncMock) as convert:
            result = await normalizer.continue_conversation_async(target=target, conversation_id="offline-continuation")
        convert.assert_not_awaited()
    assert len(requests) == 2
    assert requests[1]["input"] == [
        *requests[0]["input"],
        {"role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
    ]
    after = list(normalizer.memory.get_conversation_messages(conversation_id="offline-continuation"))
    assert len(after) == 3
    assert [message.get_piece().id for message in after[:2]] == [message.get_piece().id for message in before]
    assert json.loads(result.get_value())["call_id"] == "call-offline-1"
    assert target.supports_conversation_continuation


@pytest.mark.usefixtures("patch_central_database")
async def test_unsupported_continuation_rejects_without_fabricated_messages_async() -> None:
    from tests.unit.mocks import MockPromptTarget

    target = MockPromptTarget()
    normalizer = PromptNormalizer()
    assert not target.supports_conversation_continuation
    with pytest.raises(NotImplementedError, match="does not support"):
        await normalizer.continue_conversation_async(target=target, conversation_id="offline-unsupported")
    assert list(normalizer.memory.get_conversation_messages(conversation_id="offline-unsupported")) == []


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("failure", [RuntimeError("OFFLINE transport failed"), []])
async def test_failed_continuation_does_not_create_error_or_request_row_async(failure: RuntimeError | list) -> None:
    from pyrit.exceptions import EmptyResponseException

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=openai_response_json_dict()))
    ) as client:
        target = OpenAIResponseTarget(
            endpoint="https://fixture.invalid/v1",
            api_key="offline",
            model_name="offline-fixture",
            auto_execute_tools=False,
            httpx_client_kwargs={"http_client": client, "max_retries": 0},
        )
        normalizer = PromptNormalizer()
        await normalizer.send_prompt_async(
            message=MessagePiece(role="user", original_value="OFFLINE/SIMULATED").to_message(),
            target=target,
            conversation_id="offline-failure",
        )
        before = list(normalizer.memory.get_conversation_messages(conversation_id="offline-failure"))
        continuation = (
            AsyncMock(side_effect=failure) if isinstance(failure, RuntimeError) else AsyncMock(return_value=[])
        )
        with patch.object(target, "_continue_conversation_to_target_async", new=continuation):
            with pytest.raises((RuntimeError, EmptyResponseException)):
                await normalizer.continue_conversation_async(target=target, conversation_id="offline-failure")
    after = list(normalizer.memory.get_conversation_messages(conversation_id="offline-failure"))
    assert [item.model_dump() for item in after] == [item.model_dump() for item in before]


@pytest.mark.usefixtures("patch_central_database")
async def test_consecutive_continuations_append_only_new_authentic_responses_async() -> None:
    requests: list[dict[str, Any]] = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        body = openai_response_json_dict()
        body["output"][0]["content"][0]["text"] = f"OFFLINE response {len(requests)}"
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        target = OpenAIResponseTarget(
            endpoint="https://fixture.invalid/v1",
            api_key="offline",
            model_name="offline-fixture",
            auto_execute_tools=False,
            httpx_client_kwargs={"http_client": client, "max_retries": 0},
        )
        normalizer = PromptNormalizer()
        with pytest.raises(ValueError, match="nonempty retained"):
            await normalizer.continue_conversation_async(target=target, conversation_id="offline-empty")
        assert requests == []
        await normalizer.send_prompt_async(
            message=MessagePiece(role="user", original_value="OFFLINE input").to_message(),
            target=target,
            conversation_id="offline-consecutive",
        )
        for _ in range(2):
            await normalizer.continue_conversation_async(target=target, conversation_id="offline-consecutive")
    assert len(requests) == 3
    assert [len(item["input"]) for item in requests] == [1, 2, 3]
    assert requests[2]["input"] == [
        *requests[1]["input"],
        {"role": "assistant", "content": [{"type": "output_text", "text": "OFFLINE response 2"}]},
    ]
    messages = list(normalizer.memory.get_conversation_messages(conversation_id="offline-consecutive"))
    assert len(messages) == 4
    assert [message.get_value() for message in messages] == [
        "OFFLINE input",
        "OFFLINE response 1",
        "OFFLINE response 2",
        "OFFLINE response 3",
    ]
