# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Real OpenAI SDK 2 calls over an in-memory HTTP transport, not mock model traffic."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
from openai import AsyncOpenAI, RateLimitError

from pyrit.executor.benchmark._inspect_ghcp_adversary_capture import InspectGhcpAdversarialCapture


async def test_real_sdk_request_and_response_bytes_are_observed_once() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "object": "chat.completion",
                "model": "qwen3:1.7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Follow up"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

    sink = AsyncMock()
    capture = InspectGhcpAdversarialCapture(
        endpoint="http://127.0.0.1:11435/v1/chat/completions",
        max_body_bytes=65_536,
        sink=sink,
        transport=httpx.MockTransport(respond),
    )
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:11435/v1",
        api_key="local",
        http_client=capture.client,
        max_retries=0,
    )
    result = await client.chat.completions.create(
        model="qwen3:1.7b", messages=[{"role": "user", "content": "Propose a follow-up"}]
    )
    await capture.close_async()
    assert result.choices[0].message.content == "Follow up"
    assert str(requests[0].url) == "http://127.0.0.1:11435/v1/chat/completions"
    assert [call.args[1] for call in sink.await_args_list] == ["request", "response"]
    assert sink.await_args_list[0].args[0] == sink.await_args_list[1].args[0]
    assert json.loads(sink.await_args_list[0].args[2])["model"] == "qwen3:1.7b"
    assert sink.await_args_list[1].args[3] == 200
    assert (capture.request_count, capture.response_count) == (1, 1)


async def test_failed_upstream_request_preserves_original_bytes() -> None:
    sink = AsyncMock()
    capture = InspectGhcpAdversarialCapture(
        endpoint="http://127.0.0.1:11435/v1/chat/completions",
        max_body_bytes=65_536,
        sink=sink,
        transport=httpx.MockTransport(lambda request: httpx.Response(429, content=b'{"error":"rate limit"}')),
    )
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:11435/v1",
        api_key="local",
        http_client=capture.client,
        max_retries=0,
    )
    try:
        await client.chat.completions.create(
            model="qwen3:1.7b", messages=[{"role": "user", "content": "Benign prompt"}]
        )
        raise AssertionError("SDK accepted an actual 429 from its model source.")
    except RateLimitError:
        pass
    finally:
        await capture.close_async()
    assert sink.await_args_list[1].args[2:] == (b'{"error":"rate limit"}', 429, "host_adversarial_model_rejected")
