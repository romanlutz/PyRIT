# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Typed Inspect ModelAPI responses from exact local HTTP bytes; no network."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.tool import ToolInfo, ToolParams

from pyrit.executor.benchmark.inspect_ghcp_model import InspectLoopbackModelAPI


def _api(*, transport: httpx.MockTransport) -> InspectLoopbackModelAPI:
    return InspectLoopbackModelAPI(
        model_name="qwen3:1.7b",
        base_url="http://127.0.0.1:11435/v1",
        transport=transport,
    )


def test_rejects_wildcard_or_remote_provider_endpoints() -> None:
    for endpoint in (
        "http://0.0.0.0:11435/v1",
        "http://example.com:11435/v1",
        "https://127.0.0.1:11435/v1",
        "http://127.0.0.1:11435/another/path",
    ):
        with pytest.raises(ValueError, match="loopback-only"):
            InspectLoopbackModelAPI(model_name="qwen3:1.7b", base_url=endpoint)


async def test_original_tool_call_and_http_bytes_map_to_inspect_model_output() -> None:
    requests: list[dict[str, object]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://127.0.0.1:11435/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer local"
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "qwen3:1.7b",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26},
            },
        )

    api = _api(transport=httpx.MockTransport(reply))
    capture = AsyncMock()
    api.set_capture_sink(sink=capture)
    output, source = await api.generate(
        input=[ChatMessageUser(content="Use bash to inspect your working directory")],
        tools=[
            ToolInfo(
                name="bash",
                description="Runs a command in the isolated agent container.",
                parameters=ToolParams(properties={"command": {"type": "string"}}, required=["command"]),
            )
        ],
        tool_choice="auto",
        config=GenerateConfig(max_tokens=128, timeout=5),
    )
    assert requests[0]["model"] == "qwen3:1.7b"
    assert requests[0]["tools"][0]["function"]["name"] == "bash"
    assert requests[0]["parallel_tool_calls"] is False
    assert output.choices[0].stop_reason == "tool_calls"
    assert output.choices[0].message.tool_calls[0].arguments == {"command": "pwd"}
    assert output.usage.input_tokens == 19
    assert source.request == requests[0]
    args = capture.await_args.args
    assert json.loads(args[0]) == requests[0]
    assert json.loads(args[1])["choices"][0]["message"]["tool_calls"][0]["id"] == "call-1"
    assert args[2:] == (200, None)
    api.clear_capture_sink(sink=capture)


async def test_original_host_rejection_is_captured_before_failure() -> None:
    response = b'{"error":"model rate limited"}'
    api = _api(transport=httpx.MockTransport(lambda request: httpx.Response(429, content=response)))
    capture = AsyncMock()
    api.set_capture_sink(sink=capture)
    with pytest.raises(httpx.HTTPStatusError):
        await api.generate(
            input=[ChatMessageUser(content="harmless input")],
            tools=[],
            tool_choice="auto",
            config=GenerateConfig(max_tokens=16),
        )
    assert capture.await_args.args[1:] == (response, 429, "host_model_rejected")
