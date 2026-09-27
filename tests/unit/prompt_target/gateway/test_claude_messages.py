# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import httpx
import pytest

from pyrit.prompt_target.gateway.claude_messages import create_claude_messages_app
from pyrit.prompt_target.gateway.messages_contract import (
    MessagesCapabilities,
    MessagesCoverage,
    MessagesResponse,
    MessagesStream,
)
from pyrit.prompt_target.gateway.responses_contract import GatewayFrameKind, GatewayLimits
from tests.unit.prompt_target.gateway.messages_mocks import (
    GUEST_HEADERS,
    LIMITS,
    ROUTE,
    TOOL,
    TOOL_CALL,
    TOOL_RESULT,
    FakeMessagesBackend,
    event,
    message,
    request_body,
    response_bytes,
    text_frames,
    tool_frames,
)


def _client(
    *,
    backend: FakeMessagesBackend | None,
    limits: GatewayLimits = LIMITS,
    observer: AsyncMock | None = None,
) -> httpx.AsyncClient:
    app = create_claude_messages_app(route=ROUTE, limits=limits, backend=backend, observation_callback=observer)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid")


async def test_messages_nonstream_keeps_original_body_version_beta_and_usage_async() -> None:
    original = (
        b'{"model":"claude-offline-model", "max_tokens":16, "messages":'
        b'[{"role":"user","content":"OFFLINE"}], "system":'
        b'[{"type":"text","text":"OFFLINE system","cache_control":{"type":"ephemeral"}}]}'
    )
    provider = response_bytes()
    backend = FakeMessagesBackend(
        responses=[MessagesResponse(status_code=200, body=provider, headers=(("content-type", "application/json"),))]
    )
    observer = AsyncMock()
    headers = {
        **GUEST_HEADERS,
        "anthropic-beta": "verified-tool-beta-2026-09-01",
        "content-type": "application/json",
    }
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages?beta=true", content=original, headers=headers)
    assert reply.status_code == 200
    assert reply.content == provider
    assert reply.json()["usage"]["output_tokens"] == 3
    sent = backend.requests[0]
    assert sent.body_bytes == original
    assert sent.body == json.loads(original)
    assert sent.query_string == b"beta=true"
    assert sent.anthropic_version == "2023-06-01"
    assert sent.anthropic_beta == "verified-tool-beta-2026-09-01"
    assert not hasattr(sent, "headers")
    ingress, egress = (call.args[0] for call in observer.await_args_list)
    assert [ingress.kind, egress.kind] == [GatewayFrameKind.REQUEST, GatewayFrameKind.RESPONSE]
    assert [ingress.frame, egress.frame] == [original, provider]
    assert ingress.headers == (
        ("anthropic-version", "2023-06-01"),
        ("anthropic-beta", "verified-tool-beta-2026-09-01"),
    )
    assert ingress.query_string == egress.query_string == "beta=true"
    assert MessagesCoverage.PROMPT_CACHING in ingress.coverage
    assert MessagesCoverage.COMPLETED in egress.coverage
    assert ROUTE.guest_token.encode() not in ingress.frame


async def test_anthropic_sse_preserves_original_text_pings_and_usage_order_async() -> None:
    frames = text_frames()
    backend = FakeMessagesBackend(streams=[frames])
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages?beta=true", json=request_body(streaming=True), headers=GUEST_HEADERS)
    assert reply.status_code == 200
    assert reply.headers["content-type"].startswith("text/event-stream")
    assert reply.content == b"".join(frames)
    assert b"data: [DONE]" not in reply.content
    assert backend.closed
    observations = [call.args[0] for call in observer.await_args_list]
    assert [item.frame for item in observations[1:]] == frames
    assert all(item.kind == GatewayFrameKind.RESPONSE_EVENT for item in observations[1:])
    assert MessagesCoverage.PING in observations[2].coverage
    assert MessagesCoverage.TEXT in observations[5].coverage
    assert MessagesCoverage.COMPLETED in observations[-1].coverage


async def test_cli_tool_result_is_forwarded_unchanged_and_no_host_callback_runs_async() -> None:
    frames = tool_frames()
    backend = FakeMessagesBackend(
        streams=[frames],
        responses=[
            MessagesResponse(status_code=200, body=response_bytes(), headers=(("content-type", "application/json"),))
        ],
    )
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        first = await client.post("/v1/messages", json=request_body(streaming=True, tools=True), headers=GUEST_HEADERS)
        follow_up = await client.post(
            "/v1/messages",
            json={
                "model": ROUTE.model,
                "max_tokens": 16,
                "tools": [TOOL],
                "messages": [
                    {"role": "assistant", "content": [TOOL_CALL]},
                    {"role": "user", "content": [TOOL_RESULT]},
                ],
            },
            headers=GUEST_HEADERS,
        )
    assert first.status_code == follow_up.status_code == 200
    assert first.content == b"".join(frames)
    assert follow_up.json()["content"][0]["text"] == "OFFLINE reply"
    assert len(backend.requests) == 2
    assert backend.requests[1].body["messages"][1]["content"][0]["content"] == "exact\nOFFLINE result"
    requests = [call.args[0] for call in observer.await_args_list if call.args[0].kind == GatewayFrameKind.REQUEST]
    assert MessagesCoverage.TOOL_DEFINITION in requests[0].coverage
    assert MessagesCoverage.TOOL_RESULT in requests[1].coverage
    assert MessagesCoverage.TOOL_USE in requests[1].coverage
    assert not hasattr(backend, "tool_executor")


async def test_signed_thinking_blocks_are_forwarded_without_synthesizing_a_signature_async() -> None:
    reasoning = {"type": "thinking", "thinking": "OFFLINE reasoning", "signature": "signed-offline"}
    provider = response_bytes(content=[reasoning, {"type": "text", "text": "OFFLINE answer"}])
    backend = FakeMessagesBackend(
        responses=[MessagesResponse(status_code=200, body=provider, headers=(("content-type", "application/json"),))]
    )
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post(
            "/v1/messages",
            json={**request_body(), "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
            headers=GUEST_HEADERS,
        )
    assert reply.status_code == 200
    assert reply.content == provider
    assert backend.requests[0].body["thinking"] == {"type": "adaptive"}
    assert MessagesCoverage.THINKING in observer.await_args_list[-1].args[0].coverage


@pytest.mark.parametrize(
    ("headers", "status", "code"),
    [
        ({}, 401, "invalid_token"),
        (
            {"Authorization": "Bearer wrong", "anthropic-version": "2023-06-01", "X-PyRIT-Run-ID": ROUTE.run_id},
            401,
            "invalid_token",
        ),
        ({"Authorization": f"Bearer {ROUTE.guest_token}", "anthropic-version": "2023-06-01"}, 403, "invalid_run"),
        (
            {
                "Authorization": f"Bearer {ROUTE.guest_token}",
                "anthropic-version": "2023-06-01",
                "X-PyRIT-Run-ID": "other",
            },
            403,
            "invalid_run",
        ),
        (
            [
                ("Authorization", f"Bearer {ROUTE.guest_token}"),
                ("Authorization", f"Bearer {ROUTE.guest_token}"),
                ("X-PyRIT-Run-ID", ROUTE.run_id),
                ("anthropic-version", "2023-06-01"),
            ],
            401,
            "invalid_token",
        ),
    ],
)
async def test_routing_identity_rejects_unauthenticated_guest_before_observation_async(
    headers: dict[str, str] | list[tuple[str, str]], status: int, code: str
) -> None:
    observer = AsyncMock()
    backend = FakeMessagesBackend()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages", json=request_body(), headers=headers)
    assert reply.status_code == status
    assert reply.json()["type"] == "error"
    assert reply.json()["error"]["message"].startswith(code)
    assert ROUTE.guest_token not in reply.text
    assert backend.requests == []
    observer.assert_not_awaited()


async def test_only_documented_messages_path_and_optional_beta_query_are_allowed_async() -> None:
    backend = FakeMessagesBackend()
    async with _client(backend=backend) as client:
        for path in (
            "/v1/messages/",
            "/v1/responses",
            "/v1/messages/count_tokens",
            "/v1/messages/batches",
            "/v1/models",
            "/api/hello",
        ):
            assert (await client.post(path, json=request_body(), headers=GUEST_HEADERS)).status_code == 404
        assert (await client.head("/api/hello")).status_code == 404
        assert (await client.get("/v1/messages", headers=GUEST_HEADERS)).status_code == 405
        invalid = await client.post(
            "/v1/messages?url=https://untrusted.invalid", json=request_body(), headers=GUEST_HEADERS
        )
    assert invalid.status_code == 501
    assert invalid.json()["error"]["message"].startswith("unsupported_feature")
    assert backend.requests == []


@pytest.mark.parametrize(
    ("extra", "status"),
    [
        ({"model": "other-model"}, 400),
        ({"max_tokens": True}, 400),
        ({"max_tokens": 99_999}, 429),
        ({"context_management": {"edits": []}}, 501),
        ({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}, 501),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": {"type": "url", "url": "https://untrusted.invalid"}}],
                    }
                ]
            },
            501,
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "document", "source": {"type": "url", "url": "https://untrusted.invalid"}}
                        ],
                    }
                ]
            },
            501,
        ),
        ({"messages": [{"role": "user", "content": [{"type": "tool_reference", "tool_name": "remote"}]}]}, 501),
        ({"tools": [{**TOOL, "defer_loading": True}]}, 501),
        ({"output_config": {"format": {"type": "json_schema"}}}, 501),
        ({"thinking": {"type": "unsupported"}}, 501),
        ({"service_tier": "fast"}, 501),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "t", "content": [{"type": "image"}]}],
                    }
                ]
            },
            501,
        ),
    ],
)
async def test_unsupported_or_invalid_request_never_reaches_host_model_async(
    extra: dict[str, object], status: int
) -> None:
    backend = FakeMessagesBackend()
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages", json={**request_body(), **extra}, headers=GUEST_HEADERS)
    assert reply.status_code == status
    assert reply.json()["type"] == "error"
    assert backend.requests == []
    observer.assert_not_awaited()


@pytest.mark.parametrize(
    "options",
    [
        {"stream": True},
        {"tools": [TOOL]},
        {"thinking": {"type": "adaptive"}},
        {"output_config": {"effort": "low"}},
        {"system": [{"type": "text", "text": "OFFLINE", "cache_control": {"type": "ephemeral"}}]},
    ],
)
async def test_unverified_backend_capability_is_rejected_before_provider_io_async(
    options: dict[str, object],
) -> None:
    backend = FakeMessagesBackend(capabilities=MessagesCapabilities())
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages", json={**request_body(), **options}, headers=GUEST_HEADERS)
    assert reply.status_code == 501
    assert reply.json()["error"]["message"].startswith("unsupported_feature")
    assert backend.requests == []
    observer.assert_not_awaited()


@pytest.mark.parametrize(
    "options",
    [
        {"messages": [{"role": "user", "content": [{"type": [], "text": "OFFLINE"}]}]},
        {"tool_choice": {"type": "tool", "name": []}, "tools": [TOOL]},
        {"messages": [{"role": [], "content": "OFFLINE"}]},
        {"tools": [{**TOOL, "type": []}]},
        {"thinking": {"type": []}},
    ],
)
async def test_malformed_nested_request_fields_fail_as_explicit_client_errors_async(
    options: dict[str, object],
) -> None:
    backend = FakeMessagesBackend()
    async with _client(backend=backend) as client:
        reply = await client.post("/v1/messages", json={**request_body(), **options}, headers=GUEST_HEADERS)
    assert reply.status_code in (400, 501)
    assert reply.json()["type"] == "error"
    assert backend.requests == []


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({**GUEST_HEADERS, "x-api-key": "not-allowed"}, 501),
        ({**GUEST_HEADERS, "anthropic-workspace-id": "another-workspace"}, 501),
        ({**GUEST_HEADERS, "anthropic-version": "2030-01-01"}, 501),
        ({**GUEST_HEADERS, "anthropic-beta": "unknown-experimental-2026-09-01"}, 501),
        ({**GUEST_HEADERS, "anthropic-beta": "oauth-2026-09-01"}, 501),
        ({**GUEST_HEADERS, "content-type": "text/plain"}, 415),
        (
            [
                ("Authorization", f"Bearer {ROUTE.guest_token}"),
                ("X-PyRIT-Run-ID", ROUTE.run_id),
                ("anthropic-version", "2023-06-01"),
                ("anthropic-version", "2023-06-01"),
            ],
            400,
        ),
    ],
)
async def test_unsupported_headers_are_rejected_instead_of_forwarded_async(
    headers: dict[str, str] | list[tuple[str, str]], status: int
) -> None:
    backend = FakeMessagesBackend()
    async with _client(backend=backend) as client:
        reply = await client.post("/v1/messages", json=request_body(), headers=headers)
    assert reply.status_code == status
    assert backend.requests == []


@pytest.mark.parametrize(
    "bad",
    [
        b'{"model":"claude-offline-model","model":"claude-offline-model","max_tokens":16,"messages":[]}',
        b'{"model":"claude-offline-model","max_tokens":NaN,"messages":[]}',
        b"{invalid JSON",
        b"[]",
    ],
)
async def test_nonstandard_request_json_fails_closed_async(bad: bytes) -> None:
    backend = FakeMessagesBackend()
    async with _client(backend=backend) as client:
        reply = await client.post(
            "/v1/messages",
            content=bad,
            headers={
                **GUEST_HEADERS,
                "content-type": "application/json",
            },
        )
    assert reply.status_code == 400
    assert backend.requests == []


async def test_missing_backend_explicitly_refuses_to_fabricate_claude_output_async() -> None:
    async with _client(backend=None) as client:
        reply = await client.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert reply.status_code == 501
    assert reply.json()["error"]["message"].startswith("model_backend_required")


async def test_run_limits_reserve_output_and_input_bytes_atomically_async() -> None:
    limits = GatewayLimits(max_requests=1, max_output_tokens_per_request=16, max_total_output_tokens=16)
    backend = FakeMessagesBackend()
    async with _client(backend=backend, limits=limits) as client:
        first, second = await asyncio.gather(
            client.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS),
            client.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS),
        )
    assert sorted([first.status_code, second.status_code]) == [200, 429]
    assert len(backend.requests) == 1
    too_small = GatewayLimits(max_request_bytes=16)
    async with _client(backend=FakeMessagesBackend(), limits=too_small) as client:
        size_error = await client.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert size_error.status_code == 413
    assert size_error.json()["error"]["message"].startswith("request_too_large")


async def test_total_input_bytes_and_reserved_output_tokens_stop_later_requests_async() -> None:
    raw = json.dumps(request_body(), separators=(",", ":")).encode()
    first_response = MessagesResponse(
        status_code=200, body=response_bytes(), headers=(("content-type", "application/json"),)
    )
    byte_limits = GatewayLimits(max_total_request_bytes=2 * len(raw) - 1)
    byte_backend = FakeMessagesBackend(responses=[first_response])
    async with _client(backend=byte_backend, limits=byte_limits) as client:
        first = await client.post(
            "/v1/messages", content=raw, headers={**GUEST_HEADERS, "content-type": "application/json"}
        )
        exhausted = await client.post(
            "/v1/messages", content=raw, headers={**GUEST_HEADERS, "content-type": "application/json"}
        )
    assert first.status_code == 200
    assert exhausted.status_code == 429
    assert exhausted.json()["error"]["message"].startswith("input_byte_budget")
    assert len(byte_backend.requests) == 1

    token_limits = GatewayLimits(max_total_output_tokens=16)
    token_backend = FakeMessagesBackend(responses=[first_response])
    async with _client(backend=token_backend, limits=token_limits) as client:
        assert (await client.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)).status_code == 200
        exhausted = await client.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert exhausted.status_code == 429
    assert exhausted.json()["error"]["message"].startswith("token_budget")
    assert len(token_backend.requests) == 1


@pytest.mark.parametrize(
    "provider",
    [
        response_bytes(content=[{"type": "server_tool_use", "id": "remote"}]),
        response_bytes(
            content=[{"type": "tool_use", "id": "unknown", "name": "unadvertised", "input": {}}], stop_reason="tool_use"
        ),
        json.dumps({**message(), "usage": {"input_tokens": 9, "output_tokens": 17}}).encode(),
        json.dumps({**message(), "usage": None}).encode(),
        b"{broken JSON",
    ],
)
async def test_unsafe_provider_response_is_not_sent_as_claude_success_async(provider: bytes) -> None:
    backend = FakeMessagesBackend(
        responses=[MessagesResponse(status_code=200, body=provider, headers=(("content-type", "application/json"),))]
    )
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert reply.status_code == 502
    assert reply.json()["type"] == "error"
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert provider != reply.content


@pytest.mark.parametrize(
    "frames",
    [
        [event(name="content_block_delta", index=0, delta={"type": "text_delta", "text": "OFFLINE"})],
        text_frames()[:-1],
        [*text_frames()[:3], event(name="content_block_stop", index=0)],
        [*tool_frames()[:3], event(name="content_block_stop", index=0)],
        [
            *text_frames()[:-1],
            event(name="message_delta", delta={"stop_reason": "end_turn"}, usage={"output_tokens": 20}),
        ],
    ],
)
async def test_invalid_sse_reports_host_error_without_a_fake_message_stop_async(frames: list[bytes]) -> None:
    backend = FakeMessagesBackend(streams=[frames])
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages", json=request_body(streaming=True, tools=True), headers=GUEST_HEADERS)
    boundary = observer.await_args_list[-1].args[0]
    assert boundary.kind == GatewayFrameKind.GATEWAY_ERROR
    assert MessagesCoverage.FAILED in boundary.coverage
    assert boundary.frame in reply.content
    assert b"event: error\n" in reply.content or reply.status_code == 502
    assert b"data: [DONE]" not in reply.content
    assert backend.closed


async def test_boolean_block_index_is_not_accepted_as_integer_one_async() -> None:
    start = text_frames()[0]
    frames = [
        start,
        event(name="content_block_start", index=0, content_block={"type": "text", "text": ""}),
        event(name="content_block_stop", index=0),
        event(name="content_block_start", index=1, content_block={"type": "text", "text": ""}),
        event(name="content_block_delta", index=True, delta={"type": "text_delta", "text": "not index one"}),
    ]
    backend = FakeMessagesBackend(streams=[frames])
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages", json=request_body(streaming=True), headers=GUEST_HEADERS)
    assert reply.status_code == 200
    assert b"invalid_backend_response" in reply.content
    assert frames[4] not in reply.content
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR


async def test_empty_tool_input_delta_can_still_mean_an_empty_object_async() -> None:
    frames = tool_frames()
    frames[2] = event(name="content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": ""})
    del frames[3]
    backend = FakeMessagesBackend(streams=[frames])
    async with _client(backend=backend) as client:
        reply = await client.post("/v1/messages", json=request_body(streaming=True, tools=True), headers=GUEST_HEADERS)
    assert reply.status_code == 200
    assert reply.content == b"".join(frames)
    assert backend.closed


async def test_duplicate_tool_block_stop_is_rejected_before_a_second_tool_call_async() -> None:
    frames = tool_frames()
    duplicate = event(name="content_block_stop", index=0)
    frames.insert(5, duplicate)
    backend = FakeMessagesBackend(streams=[frames])
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages", json=request_body(streaming=True, tools=True), headers=GUEST_HEADERS)
    assert reply.status_code == 200
    assert reply.content.count(frames[4]) == 1
    assert duplicate not in reply.content[len(b"".join(frames[:5])) :]
    assert b"invalid_backend_response" in reply.content
    assert b"event: message_stop" not in reply.content
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR


async def test_genuine_provider_error_bytes_and_retry_header_are_not_repackaged_async() -> None:
    original = b'{"type":"error","error":{"type":"rate_limit_error","message":"provider throttle"}}'
    backend = FakeMessagesBackend(
        responses=[
            MessagesResponse(
                status_code=429,
                body=original,
                headers=(("content-type", "application/json"), ("retry-after", "2"), ("x-should-retry", "true")),
            )
        ]
    )
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert reply.status_code == 429
    assert reply.content == original
    assert reply.headers["retry-after"] == "2"
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.RESPONSE
    assert observer.await_args_list[-1].args[0].frame == original
    assert MessagesCoverage.FAILED in observer.await_args_list[-1].args[0].coverage


@pytest.mark.parametrize(
    ("status", "headers"),
    [
        (200, (("Authorization", "secret-from-provider"), ("content-type", "application/json"))),
        (200, ()),
        (201, (("content-type", "application/json"),)),
        (429, (("content-type", "text/html"),)),
    ],
)
async def test_fake_backend_with_invalid_status_or_headers_fails_explicitly_async(
    status: int, headers: tuple[tuple[str, str], ...]
) -> None:
    backend = FakeMessagesBackend(
        responses=[MessagesResponse(status_code=status, body=response_bytes(), headers=headers)]
    )
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert reply.status_code == 502
    assert reply.json()["error"]["message"].startswith("invalid_backend_response")
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert "secret-from-provider" not in reply.text
    assert all("secret-from-provider" not in str(call.args[0].headers) for call in observer.await_args_list)


async def test_post_stream_observer_failure_yields_host_error_not_completion_async() -> None:
    backend = FakeMessagesBackend(streams=[text_frames()])
    observer = AsyncMock(side_effect=[None, None, RuntimeError("offline recorder error"), None])
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages", json=request_body(streaming=True), headers=GUEST_HEADERS)
    assert reply.status_code == 200
    assert b"event: error\n" in reply.content
    assert b"observation_failed" in reply.content
    assert b"event: message_stop" not in reply.content
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert backend.closed


async def test_provider_failure_survives_recorder_failure_without_claiming_success_async(
    caplog: pytest.LogCaptureFixture,
) -> None:
    original = b'{"type":"error","error":{"type":"rate_limit_error","message":"provider offline"}}'
    backend = FakeMessagesBackend(
        responses=[MessagesResponse(status_code=429, body=original, headers=(("content-type", "application/json"),))]
    )
    observer = AsyncMock(side_effect=[None, RuntimeError("private recorder detail")])
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert reply.status_code == 429
    assert reply.content == original
    assert "private recorder detail" not in reply.text + caplog.text
    assert "observation_failed" in caplog.text
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.RESPONSE


async def test_unexpected_host_backend_exception_is_sanitized_and_observed_async() -> None:
    class BrokenBackend(FakeMessagesBackend):
        async def create_message_async(self, *, request: object) -> MessagesResponse:
            raise RuntimeError("private host detail")

    backend = BrokenBackend()
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert reply.status_code == 502
    assert reply.json()["error"]["message"].startswith("backend_failed")
    assert "private host detail" not in reply.text
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR


async def test_terminal_error_observer_failure_keeps_primary_incomplete_stream_async(
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = text_frames()[0]
    backend = FakeMessagesBackend(streams=[[first]])
    observer = AsyncMock(side_effect=[None, None, RuntimeError("private recorder detail")])
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/messages", json=request_body(streaming=True), headers=GUEST_HEADERS)
    assert reply.status_code == 200
    assert reply.content.startswith(first)
    assert b"incomplete_stream" in reply.content
    assert b"event: message_stop" not in reply.content
    assert "private recorder detail" not in reply.text + caplog.text
    assert "incomplete_stream" in caplog.text
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert backend.closed


async def test_cancelling_a_guest_stream_closes_model_without_refunding_budget_async() -> None:
    class HangingBackend(FakeMessagesBackend):
        def __init__(self) -> None:
            super().__init__()
            self.waiting = asyncio.Event()
            self.finished = asyncio.Event()

        async def open_stream_async(self, *, request: object) -> MessagesStream:
            self.requests.append(request)
            return MessagesStream(
                frames=self._hang_async(),
                headers=(("content-type", "text/event-stream"),),
                _close=self._close_async,
            )

        async def _hang_async(self) -> AsyncGenerator[bytes, None]:
            try:
                yield text_frames()[0]
                self.waiting.set()
                await asyncio.Event().wait()
            finally:
                self.finished.set()

    backend = HangingBackend()
    async with _client(backend=backend, limits=GatewayLimits(max_requests=1)) as client:
        pending = asyncio.create_task(
            client.post("/v1/messages", json=request_body(streaming=True), headers=GUEST_HEADERS)
        )
        try:
            await asyncio.wait_for(backend.waiting.wait(), timeout=2)
        finally:
            pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(backend.finished.wait(), timeout=2)
        exhausted = await client.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert exhausted.status_code == 429
    assert backend.closed


def test_messages_route_policy_rejects_oauth_and_untyped_capabilities() -> None:
    with pytest.raises(ValueError, match="OAuth"):
        MessagesCapabilities(allowed_beta_values=frozenset({"oauth-client-beta-2026-09-01"}))
    with pytest.raises(ValueError, match="explicitly verified boolean"):
        MessagesCapabilities(streaming="true")
    with pytest.raises(ValueError, match="allowed_beta_values"):
        MessagesCapabilities(allowed_beta_values={"verified-tool-beta-2026-09-01"})
    assert ROUTE.guest_token not in repr(ROUTE)
