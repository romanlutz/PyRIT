# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import httpx
import pytest

from pyrit.prompt_target.gateway.claude_messages import create_claude_messages_app
from pyrit.prompt_target.gateway.httpx_messages_backend import HttpxMessagesBackend
from pyrit.prompt_target.gateway.messages_contract import (
    MessagesBackendError,
    MessagesBackendErrorCode,
    MessagesCapabilities,
    MessagesCoverage,
    MessagesRequest,
)
from pyrit.prompt_target.gateway.responses_contract import GatewayFrameKind, GatewayLimits
from tests.unit.prompt_target.gateway.messages_mocks import (
    CAPABILITIES,
    GUEST_HEADERS,
    HOST_KEY,
    LIMITS,
    ROUTE,
    TOOL,
    TOOL_CALL,
    TOOL_RESULT,
    event,
    message,
    request_body,
    response_bytes,
    text_frames,
    tool_frames,
)

ENDPOINT = "https://model.invalid/v1/messages"


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(
        self,
        *,
        chunks: list[bytes],
        failure: Exception | None = None,
        pause_seconds: float = 0,
        wait_forever: bool = False,
    ) -> None:
        self._chunks = chunks
        self._failure = failure
        self._pause_seconds = pause_seconds
        self._wait_forever = wait_forever
        self.waiting = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        self.waiting.set()
        if self._pause_seconds:
            await asyncio.sleep(self._pause_seconds)
        if self._wait_forever:
            await asyncio.Event().wait()
        if self._failure:
            raise self._failure

    async def aclose(self) -> None:
        self.closed = True


def _backend(
    *,
    client: httpx.AsyncClient,
    limits: GatewayLimits = LIMITS,
    capabilities: MessagesCapabilities = CAPABILITIES,
    timeout_seconds: float | None = None,
) -> HttpxMessagesBackend:
    return HttpxMessagesBackend(
        route=ROUTE,
        endpoint=ENDPOINT,
        host_api_key=HOST_KEY,
        client=client,
        limits=limits,
        capabilities=capabilities,
        timeout_seconds=timeout_seconds,
    )


def _request(*, streaming: bool = False, beta: str | None = None) -> MessagesRequest:
    body = request_body(streaming=streaming, tools=streaming)
    raw = json.dumps(body, separators=(",", ":")).encode()
    return MessagesRequest(
        run_id=ROUTE.run_id,
        request_id="offline-request-1",
        body_bytes=raw,
        body=body,
        anthropic_version="2023-06-01",
        anthropic_beta=beta,
        query_string=b"beta=true",
        max_tokens=16,
        streaming=streaming,
        advertised_tools=frozenset({"shell_command"}) if streaming else frozenset(),
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://model.invalid/v1/messages",
        "https://model.invalid/v1/messages?beta=true",
        "https://model.invalid/v1/messages#fragment",
        "https://guest:password@model.invalid/v1/messages",
        "https://model.invalid/v1//messages",
        "https://model.invalid/v1/../v1/messages",
        "https://model.invalid/v1/messages%2F",
        "https://model.invalid/v1/responses",
        "https://model.invalid:99999/v1/messages",
    ],
)
async def test_backend_requires_fixed_host_https_messages_endpoint_async(endpoint: str) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
        with pytest.raises(ValueError, match="pinned HTTPS"):
            HttpxMessagesBackend(
                route=ROUTE,
                endpoint=endpoint,
                host_api_key=HOST_KEY,
                client=client,
                limits=LIMITS,
                capabilities=CAPABILITIES,
            )


async def test_host_key_and_capabilities_must_be_explicit_and_distinct_from_guest_async() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
        with pytest.raises(ValueError, match="Host API key"):
            HttpxMessagesBackend(
                route=ROUTE,
                endpoint=ENDPOINT,
                host_api_key=ROUTE.guest_token,
                client=client,
                limits=LIMITS,
                capabilities=CAPABILITIES,
            )
        with pytest.raises(ValueError, match="Explicit Messages capabilities"):
            HttpxMessagesBackend(
                route=ROUTE,
                endpoint=ENDPOINT,
                host_api_key=HOST_KEY,
                client=client,
                limits=LIMITS,
                capabilities=None,
            )
        with pytest.raises(ValueError, match="Upstream timeout"):
            _backend(client=client, timeout_seconds=LIMITS.timeout_seconds + 1)


async def test_backend_validates_run_original_bytes_and_betas_before_http_async() -> None:
    requests: list[httpx.Request] = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=response_bytes(), headers={"Content-Type": "application/json"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        backend = _backend(client=client)
        wrong_run = _request()
        wrong_run = MessagesRequest(**{**vars(wrong_run), "run_id": "wrong-run"})
        with pytest.raises(ValueError, match="host-owned run"):
            await backend.create_message_async(request=wrong_run)
        mismatch = _request()
        mismatch.body["model"] = "changed-after-observation"
        with pytest.raises(ValueError, match="original wire bytes"):
            await backend.create_message_async(request=mismatch)
        unknown_beta = _request(beta="unverified-capability")
        with pytest.raises(ValueError, match="verified capabilities"):
            await backend.create_message_async(request=unknown_beta)
    assert requests == []


async def test_nonstream_real_host_api_key_is_injected_without_guest_auth_or_client_defaults_async() -> None:
    original = response_bytes()
    requests: list[httpx.Request] = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=original, headers={"Content-Type": "application/json"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(provider),
        follow_redirects=True,
        headers={"x-api-key": "wrong-default-key", "authorization": "Bearer wrong-default"},
        auth=httpx.BasicAuth("wrong", "default"),
    ) as client:
        result = await _backend(client=client).create_message_async(request=_request())
    assert result.status_code == 200
    assert result.body == original
    assert len(requests) == 1
    outgoing = requests[0]
    assert str(outgoing.url) == ENDPOINT + "?beta=true"
    assert outgoing.method == "POST"
    assert outgoing.headers["x-api-key"] == HOST_KEY
    assert "authorization" not in outgoing.headers
    assert outgoing.headers["anthropic-version"] == "2023-06-01"
    assert "anthropic-beta" not in outgoing.headers
    assert outgoing.content == _request().body_bytes
    assert ROUTE.guest_token not in str(outgoing.headers)
    assert "x-pyrit-run-id" not in outgoing.headers
    assert HOST_KEY.encode() not in outgoing.content


async def test_beta_header_and_raw_request_bytes_survive_full_asgi_http_round_trip_async() -> None:
    raw = (
        b'{ "model":"claude-offline-model", "max_tokens":16,'
        b' "messages":[{"role":"user","content":"OFFLINE"}],'
        b' "system":[{"type":"text","text":"OFFLINE system","cache_control":{"type":"ephemeral"}}]}'
    )
    upstream: list[httpx.Request] = []
    observer = AsyncMock()
    original = response_bytes()

    def provider(request: httpx.Request) -> httpx.Response:
        upstream.append(request)
        return httpx.Response(
            200,
            content=original,
            headers={"Content-Type": "application/json", "Anthropic-Ratelimit-Unified-Status": "offline"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/messages?beta=true",
                content=raw,
                headers={
                    **GUEST_HEADERS,
                    "Content-Type": "application/json",
                    "anthropic-beta": "verified-tool-beta-2026-09-01",
                },
            )
    assert reply.status_code == 200
    assert reply.content == original
    assert reply.headers["Anthropic-Ratelimit-Unified-Status"] == "offline"
    assert len(upstream) == 1
    assert upstream[0].content == raw
    assert upstream[0].headers["anthropic-beta"] == "verified-tool-beta-2026-09-01"
    assert upstream[0].headers["x-api-key"] == HOST_KEY
    records = [call.args[0] for call in observer.await_args_list]
    assert records[0].frame == raw
    assert records[1].frame == original
    assert records[0].headers == (
        ("anthropic-version", "2023-06-01"),
        ("anthropic-beta", "verified-tool-beta-2026-09-01"),
    )
    assert all(HOST_KEY.encode() not in record.frame for record in records)
    assert all("x-api-key" not in str(record.headers).lower() for record in records)


async def test_guest_prompt_url_is_only_text_not_an_http_destination_async() -> None:
    upstream: list[httpx.Request] = []

    def provider(request: httpx.Request) -> httpx.Response:
        upstream.append(request)
        return httpx.Response(200, content=response_bytes(), headers={"content-type": "application/json"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as host_client:
        app = create_claude_messages_app(route=ROUTE, limits=LIMITS, backend=_backend(client=host_client))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/messages",
                json={
                    **request_body(),
                    "messages": [{"role": "user", "content": "https://untrusted.invalid/guest-resource"}],
                },
                headers=GUEST_HEADERS,
            )
    assert reply.status_code == 200
    assert len(upstream) == 1
    assert str(upstream[0].url) == ENDPOINT
    assert "https://untrusted.invalid/guest-resource" in str(upstream[0].content)


async def test_streaming_text_comments_and_tool_events_relay_original_sse_chunks_async() -> None:
    frames = tool_frames()
    frames.insert(1, b": model keep-alive\r\n\r\n")
    combined = b"".join(frames)
    chunks = [combined[:1], combined[1:27], combined[27:59], combined[59:71], combined[71:]]
    stream = _ChunkStream(chunks=chunks)
    upstream: list[httpx.Request] = []
    observer = AsyncMock()

    def provider(request: httpx.Request) -> httpx.Response:
        upstream.append(request)
        return httpx.Response(
            200,
            stream=stream,
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/messages?beta=true", json=request_body(streaming=True, tools=True), headers=GUEST_HEADERS
            )
    assert reply.status_code == 200
    assert reply.content == combined
    assert b"data: [DONE]" not in reply.content
    assert stream.closed
    assert upstream[0].headers["accept"] == "text/event-stream"
    assert upstream[0].headers["x-api-key"] == HOST_KEY
    assert json.loads(upstream[0].content)["tools"] == [TOOL]
    events = [call.args[0] for call in observer.await_args_list if call.args[0].kind == GatewayFrameKind.RESPONSE_EVENT]
    assert [item.frame for item in events] == frames
    assert MessagesCoverage.PING in events[1].coverage
    assert MessagesCoverage.TOOL_USE in events[-3].coverage
    assert MessagesCoverage.COMPLETED in events[-1].coverage


async def test_signature_deltas_and_cache_usage_remain_provider_original_async() -> None:
    start = {**message(content=[]), "stop_reason": None}
    start["usage"] = {"input_tokens": 2, "output_tokens": 1, "cache_read_input_tokens": 7}
    frames = [
        event(name="message_start", message=start),
        event(name="content_block_start", index=0, content_block={"type": "thinking", "thinking": ""}),
        event(name="content_block_delta", index=0, delta={"type": "thinking_delta", "thinking": "OFFLINE reasoning"}),
        event(name="content_block_delta", index=0, delta={"type": "signature_delta", "signature": "signed-offline"}),
        event(name="content_block_stop", index=0),
        event(name="content_block_start", index=1, content_block={"type": "text", "text": ""}),
        event(name="content_block_delta", index=1, delta={"type": "text_delta", "text": "OFFLINE result"}),
        event(name="content_block_stop", index=1),
        event(name="message_delta", delta={"stop_reason": "end_turn"}, usage={"output_tokens": 12}),
        event(name="message_stop"),
    ]
    stream = _ChunkStream(chunks=[b"".join(frames)])
    observer = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        )
    ) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/messages",
                json={**request_body(streaming=True), "thinking": {"type": "adaptive"}},
                headers=GUEST_HEADERS,
            )
    assert reply.status_code == 200
    assert reply.content == b"".join(frames)
    observed_events = [
        call.args[0] for call in observer.await_args_list if call.args[0].kind == GatewayFrameKind.RESPONSE_EVENT
    ]
    assert [item.frame for item in observed_events] == frames
    assert MessagesCoverage.THINKING in observed_events[3].coverage
    assert observed_events[0].frame == frames[0]
    assert stream.closed


async def test_cli_tool_result_reaches_upstream_model_unchanged_in_next_request_async() -> None:
    stream = _ChunkStream(chunks=[b"".join(tool_frames())])
    requests: list[httpx.Request] = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(200, content=response_bytes(), headers={"Content-Type": "application/json"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as host_client:
        app = create_claude_messages_app(route=ROUTE, limits=LIMITS, backend=_backend(client=host_client))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            initial = await guest.post(
                "/v1/messages", json=request_body(streaming=True, tools=True), headers=GUEST_HEADERS
            )
            follow_up = await guest.post(
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
    assert initial.status_code == follow_up.status_code == 200
    assert len(requests) == 2
    assert json.loads(requests[1].content)["messages"][-1]["content"][0] == TOOL_RESULT
    assert stream.closed


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_genuine_provider_error_status_body_and_retry_headers_are_forwarded_async(status: int) -> None:
    error = b'{"type":"error","error":{"type":"invalid_request_error","message":"provider capability rejected"}}'
    requests: list[httpx.Request] = []
    observer = AsyncMock()

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status,
            content=error,
            headers={"Content-Type": "application/json", "retry-after": "2", "x-should-retry": "false"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert reply.status_code == status
    assert reply.content == error
    assert reply.headers["retry-after"] == "2"
    assert reply.headers["x-should-retry"] == "false"
    assert len(requests) == 1
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.RESPONSE
    assert observer.await_args_list[-1].args[0].frame == error
    assert HOST_KEY.encode() not in reply.content


async def test_stream_request_can_receive_real_provider_http_error_body_async() -> None:
    original = b'{"type":"error","error":{"type":"rate_limit_error","message":"upstream retry later"}}'
    observer = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                429, content=original, headers={"Content-Type": "application/json", "retry-after": "3"}
            )
        )
    ) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(streaming=True), headers=GUEST_HEADERS)
    assert reply.status_code == 429
    assert reply.content == original
    assert reply.headers["retry-after"] == "3"
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.RESPONSE
    assert observer.await_args_list[-1].args[0].frame == original


async def test_upstream_redirect_is_not_followed_or_relayed_as_model_success_async() -> None:
    requests: list[httpx.Request] = []
    observer = AsyncMock()

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            302,
            content=b"<html>Moved</html>",
            headers={"Location": "https://untrusted.invalid/v1/messages", "Content-Type": "text/html"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider), follow_redirects=True) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert len(requests) == 1
    assert str(requests[0].url) == ENDPOINT
    assert reply.status_code == 502
    assert b"untrusted.invalid" not in reply.content
    assert reply.headers.get("location") is None
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        ({"Content-Type": "text/plain"}, b"private upstream text"),
        ({"Content-Type": "application/json", "Content-Encoding": "gzip"}, b"not-gzip"),
        ({"Content-Type": "application/json"}, b'{"type":"message","content":"broken"}'),
        ({"Content-Type": "application/json"}, b'{"type":"message","usage":{"output_tokens":999999}}'),
        (
            {"Content-Type": "application/json"},
            json.dumps(message(content=[{"type": "text", "text": HOST_KEY}])).encode(),
        ),
        (
            {"Content-Type": "application/json"},
            b'{"type":"error","error":{"type":"invalid_request_error","message":'
            + json.dumps(HOST_KEY).encode()
            + b"}}",
        ),
    ],
)
async def test_invalid_or_credential_echo_upstream_body_never_reaches_guest_or_observer_async(
    headers: dict[str, str], body: bytes
) -> None:
    observer = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body, headers=headers))
    ) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert reply.status_code == 502
    assert HOST_KEY not in reply.text
    assert reply.json()["type"] == "error"
    assert all(HOST_KEY.encode() not in call.args[0].frame for call in observer.await_args_list)
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR


async def test_provider_error_echoing_host_auth_is_sanitized_instead_of_forwarded_async() -> None:
    upstream_error = json.dumps(
        {"type": "error", "error": {"type": "authentication_error", "message": HOST_KEY}}
    ).encode()
    observer = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(401, content=upstream_error, headers={"Content-Type": "application/json"})
        )
    ) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert reply.status_code == 502
    assert HOST_KEY not in reply.text
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert all(HOST_KEY.encode() not in call.args[0].frame for call in observer.await_args_list)


async def test_host_key_split_between_error_fields_is_not_forwarded_async() -> None:
    midpoint = len(HOST_KEY) // 2
    body = json.dumps(
        {
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": HOST_KEY[:midpoint],
                "trace": HOST_KEY[midpoint:],
            },
        }
    ).encode()
    observer = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(401, content=body, headers={"Content-Type": "application/json"})
        )
    ) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert reply.status_code == 502
    assert reply.content != body
    assert HOST_KEY not in reply.text
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR


async def test_host_key_split_between_forwardable_headers_is_not_observed_async() -> None:
    midpoint = len(HOST_KEY) // 2
    headers = {
        "Content-Type": "application/json",
        "Anthropic-Ratelimit-Unified-A": HOST_KEY[:midpoint],
        "Anthropic-Ratelimit-Unified-B": HOST_KEY[midpoint:],
    }
    observer = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=response_bytes(), headers=headers))
    ) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert reply.status_code == 502
    assert HOST_KEY not in reply.text + str(reply.headers)
    assert all(HOST_KEY not in str(call.args[0].headers) for call in observer.await_args_list)
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR


async def test_upstream_connect_failure_is_sanitized_and_observed_async() -> None:
    observer = AsyncMock()

    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"private provider detail {HOST_KEY}", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert reply.status_code == 502
    assert reply.json()["error"]["message"].startswith("upstream_network_error")
    assert observer.await_args_list[-1].args[0].error_code == "upstream_network_error"
    assert HOST_KEY not in reply.text


@pytest.mark.parametrize("declared_length", [None, 1])
async def test_backend_enforces_actual_nonstream_response_byte_ceiling_async(declared_length: int | None) -> None:
    raw = response_bytes()
    limits = GatewayLimits(max_response_bytes=len(raw) - 1)
    stream = _ChunkStream(chunks=[raw[:20], raw[20:]])
    headers = {"Content-Type": "application/json"}
    if declared_length is not None:
        headers["Content-Length"] = str(declared_length)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream, headers=headers))
    ) as host_client:
        backend = _backend(client=host_client, limits=limits)
        with pytest.raises(MessagesBackendError) as caught:
            await backend.create_message_async(request=_request())
    assert caught.value.code is MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR
    assert stream.closed


async def test_stream_response_byte_ceiling_fails_after_safe_first_frame_async() -> None:
    frames = text_frames()
    limits = GatewayLimits(max_response_bytes=len(frames[0]) + len(frames[1]) + 1)
    stream = _ChunkStream(chunks=[frames[0], frames[1], frames[2]])
    observer = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        )
    ) as host_client:
        app = create_claude_messages_app(
            route=ROUTE,
            limits=limits,
            backend=_backend(client=host_client, limits=limits),
            observation_callback=observer,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(streaming=True), headers=GUEST_HEADERS)
    assert reply.status_code == 200
    assert reply.content.startswith(frames[0])
    assert b"event: error\n" in reply.content
    assert frames[2] not in reply.content
    assert b"event: message_stop" not in reply.content
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert stream.closed


async def test_split_secret_in_streamed_text_is_held_then_rejected_async() -> None:
    frames = text_frames()
    midpoint = len(HOST_KEY) // 2
    frames[4] = event(
        name="content_block_delta",
        index=0,
        delta={"type": "text_delta", "text": HOST_KEY[:midpoint]},
    )
    frames.insert(
        5,
        event(
            name="content_block_delta",
            index=0,
            delta={"type": "text_delta", "text": HOST_KEY[midpoint:]},
        ),
    )
    stream = _ChunkStream(chunks=[frames[0], *frames[1:]])
    observer = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        )
    ) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(streaming=True), headers=GUEST_HEADERS)
    assert reply.status_code == 200
    assert reply.content.startswith(frames[0])
    assert b"event: error\n" in reply.content
    assert frames[4] not in reply.content
    assert HOST_KEY.encode() not in reply.content
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert stream.closed


async def test_json_escaped_host_api_key_is_rejected_before_sse_observation_async() -> None:
    frames = text_frames()
    escaped = HOST_KEY.replace("h", "\\u0068", 1)
    raw_json = f'{{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":"{escaped}"}}}}'
    offending = b"event: content_block_delta\ndata: " + raw_json.encode() + b"\n\n"
    frames[4] = offending
    stream = _ChunkStream(chunks=[b"".join(frames)])
    observer = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        )
    ) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(streaming=True), headers=GUEST_HEADERS)
    assert reply.status_code == 200
    assert offending not in reply.content
    assert b"event: error\n" in reply.content
    assert HOST_KEY.encode() not in reply.content
    assert all(offending != call.args[0].frame for call in observer.await_args_list)
    assert stream.closed


@pytest.mark.parametrize("kind", ["truncated", "network", "timeout"])
async def test_partial_stream_failures_produce_host_error_without_fabricated_message_stop_async(kind: str) -> None:
    first = text_frames()[0]
    error = httpx.ReadError(f"offline failure {HOST_KEY}") if kind == "network" else None
    stream = _ChunkStream(chunks=[first], failure=error, pause_seconds=0.1 if kind == "timeout" else 0)
    observer = AsyncMock()
    limits = GatewayLimits(max_request_bytes=8192, max_response_bytes=16_384, timeout_seconds=1.0)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        )
    ) as host_client:
        app = create_claude_messages_app(
            route=ROUTE,
            limits=limits,
            backend=_backend(client=host_client, limits=limits, timeout_seconds=0.03 if kind == "timeout" else None),
            observation_callback=observer,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(streaming=True), headers=GUEST_HEADERS)
    assert reply.status_code == 200
    assert reply.content.startswith(first)
    assert b"event: error\n" in reply.content
    assert b"event: message_stop\n" not in reply.content
    assert HOST_KEY.encode() not in reply.content
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert stream.closed


async def test_timed_out_dispatch_reports_typed_error_before_any_model_frame_async() -> None:
    observer = AsyncMock()
    limits = GatewayLimits(timeout_seconds=1.0)

    async def slow_provider(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.1)
        return httpx.Response(200, content=response_bytes(), headers={"Content-Type": "application/json"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(slow_provider)) as host_client:
        app = create_claude_messages_app(
            route=ROUTE,
            limits=limits,
            backend=_backend(client=host_client, limits=limits, timeout_seconds=0.01),
            observation_callback=observer,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(), headers=GUEST_HEADERS)
    assert reply.status_code == 504
    assert reply.json()["error"]["message"].startswith("upstream_timeout")
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert observer.await_args_list[-1].args[0].status_code == 504


async def test_provider_stream_error_event_is_forwarded_as_genuine_not_host_generated_async() -> None:
    error_event = event(name="error", error={"type": "overloaded_error", "message": "provider overloaded"})
    frames = [text_frames()[0], error_event]
    stream = _ChunkStream(chunks=[b"".join(frames)])
    observer = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        )
    ) as host_client:
        app = create_claude_messages_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post("/v1/messages", json=request_body(streaming=True), headers=GUEST_HEADERS)
    assert reply.status_code == 200
    assert reply.content == b"".join(frames)
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.RESPONSE_EVENT
    assert MessagesCoverage.FAILED in observer.await_args_list[-1].args[0].coverage
    assert stream.closed


async def test_cancelled_guest_request_closes_live_provider_stream_async() -> None:
    stream = _ChunkStream(chunks=[text_frames()[0]], wait_forever=True)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        )
    ) as host_client:
        app = create_claude_messages_app(route=ROUTE, limits=LIMITS, backend=_backend(client=host_client))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            pending = asyncio.create_task(
                guest.post("/v1/messages", json=request_body(streaming=True), headers=GUEST_HEADERS)
            )
            try:
                await asyncio.wait_for(stream.waiting.wait(), timeout=2)
            finally:
                pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
    assert stream.closed
