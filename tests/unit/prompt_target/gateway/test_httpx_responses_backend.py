# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from pyrit.prompt_target.gateway.codex_responses import create_codex_responses_app
from pyrit.prompt_target.gateway.httpx_responses_backend import HttpxResponsesBackend
from pyrit.prompt_target.gateway.responses_contract import (
    BackendCapabilities,
    GatewayCoverage,
    GatewayFrameKind,
    GatewayLimits,
    GatewayRoute,
    ModelBackendError,
    ModelBackendErrorCode,
    ModelRequest,
)

ROUTE = GatewayRoute(run_id="offline-run-1", model="offline-model", guest_token="guest-only-" + "x" * 32)
HOST_TOKEN = "host-only-" + "y" * 32
ENDPOINT = "https://model.invalid/v1/responses"
GUEST_HEADERS = {"Authorization": f"Bearer {ROUTE.guest_token}", "X-PyRIT-Run-ID": ROUTE.run_id}
CAPABILITIES = BackendCapabilities(streaming=True, function_tools=True, custom_tools=True, reasoning=True)
LIMITS = GatewayLimits(max_request_bytes=4_096, max_response_bytes=8_192, max_output_tokens_per_request=32)
FUNCTION = {"type": "function", "name": "shell_command", "parameters": {"type": "object"}}
CUSTOM_TOOL = {"type": "custom", "name": "apply_patch", "format": {"type": "text"}}
CALL = {
    "type": "function_call",
    "name": "shell_command",
    "call_id": "call-1",
    "arguments": '{"command":"echo OFFLINE"}',
}


def _request(*, streaming: bool = False, input_items: str | list[dict[str, Any]] = "OFFLINE") -> ModelRequest:
    body: dict[str, Any] = {"model": ROUTE.model, "input": input_items, "max_output_tokens": 16, "store": False}
    if streaming:
        body.update(stream=True, tools=[FUNCTION])
    return ModelRequest(run_id=ROUTE.run_id, request_id="req-offline", body=body, output_token_limit=16)


def _response(*, output: list[dict[str, Any]] | None = None) -> bytes:
    return json.dumps(
        {
            "id": "resp-offline",
            "object": "response",
            "model": ROUTE.model,
            "status": "completed",
            "output": output
            if output is not None
            else [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OFFLINE"}]}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        },
        separators=(",", ":"),
    ).encode()


def _event(*, name: str, sequence: int, **details: Any) -> bytes:
    content = json.dumps({"type": name, "sequence_number": sequence, **details}, separators=(",", ":"))
    return f"event: {name}\ndata: {content}\n\n".encode()


def _frames(*, output: list[dict[str, Any]] | None = None) -> list[bytes]:
    in_progress = {"id": "resp-offline", "object": "response", "model": ROUTE.model, "status": "in_progress"}
    return [
        _event(name="response.created", sequence=0, response=in_progress),
        _event(name="response.completed", sequence=1, response=json.loads(_response(output=output))),
        b"data: [DONE]\n\n",
    ]


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(
        self,
        *,
        chunks: list[bytes],
        failure: Exception | None = None,
        delay_seconds: float = 0.0,
        wait_forever: bool = False,
    ) -> None:
        self._chunks = chunks
        self._failure = failure
        self._delay_seconds = delay_seconds
        self._wait_forever = wait_forever
        self.waiting = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        self.waiting.set()
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
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
    capabilities: BackendCapabilities = CAPABILITIES,
    timeout_seconds: float | None = None,
) -> HttpxResponsesBackend:
    return HttpxResponsesBackend(
        route=ROUTE,
        endpoint=ENDPOINT,
        auth_token=HOST_TOKEN,
        client=client,
        limits=limits,
        capabilities=capabilities,
        timeout_seconds=timeout_seconds,
    )


def test_backend_error_has_no_provider_detail() -> None:
    error = ModelBackendError(code=ModelBackendErrorCode.UPSTREAM_NETWORK_ERROR)
    assert str(error) == "Unable to reach the upstream model"
    assert error.status_code == 502
    assert error.code.value == "upstream_network_error"
    assert HOST_TOKEN not in repr(error)
    with pytest.raises(ValueError, match="Unsupported model backend error code"):
        ModelBackendError(code="upstream_timeout")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://model.invalid/v1/responses",
        "https://model.invalid/v1/chat/completions",
        "https://model.invalid/v1/../responses",
        "https://model.invalid/v1//responses",
        "https://model.invalid/v1/responses?url=https://other.invalid",
        "https://model.invalid/v1/responses#fragment",
        "https://user:pass@model.invalid/v1/responses",
        "https://model.invalid:99999/v1/responses",
    ],
)
async def test_endpoint_must_be_pinned_https_without_credentials_or_query_async(endpoint: str) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
        with pytest.raises(ValueError, match="pinned HTTPS Responses endpoint"):
            HttpxResponsesBackend(
                route=ROUTE,
                endpoint=endpoint,
                auth_token=HOST_TOKEN,
                client=client,
                limits=LIMITS,
                capabilities=CAPABILITIES,
            )


async def test_backend_rejects_guest_auth_reuse_and_unknown_capabilities_async() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
        with pytest.raises(ValueError, match="Host model auth"):
            HttpxResponsesBackend(
                route=ROUTE,
                endpoint=ENDPOINT,
                auth_token=ROUTE.guest_token,
                client=client,
                limits=LIMITS,
                capabilities=CAPABILITIES,
            )
        with pytest.raises(ValueError, match="Explicit boolean model backend capabilities"):
            HttpxResponsesBackend(
                route=ROUTE, endpoint=ENDPOINT, auth_token=HOST_TOKEN, client=client, limits=LIMITS, capabilities=None
            )
        with pytest.raises(ValueError, match="Explicit boolean model backend capabilities"):
            _backend(client=client, capabilities=BackendCapabilities(streaming="yes"))
        with pytest.raises(ValueError, match="Upstream timeout"):
            _backend(client=client, timeout_seconds=LIMITS.timeout_seconds + 1)


async def test_backend_rejects_unrouted_or_unadvertised_requests_before_http_async() -> None:
    upstream_requests: list[httpx.Request] = []

    def provider(request: httpx.Request) -> httpx.Response:
        upstream_requests.append(request)
        return httpx.Response(200, content=_response(), headers={"Content-Type": "application/json"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        backend = _backend(client=client, capabilities=BackendCapabilities())
        wrong_run = _request()
        wrong_run = ModelRequest(
            run_id="other-run", request_id=wrong_run.request_id, body=wrong_run.body, output_token_limit=16
        )
        with pytest.raises(ValueError, match="host-owned run route"):
            await backend.create_response_async(request=wrong_run)

        unknown_field = _request()
        unknown_field.body["upstream_url"] = "https://untrusted.invalid"
        with pytest.raises(ValueError, match="gateway's supported Responses subset"):
            await backend.create_response_async(request=unknown_field)

        with pytest.raises(NotImplementedError, match="does not advertise"):
            await anext(backend.stream_response_async(request=_request(streaming=True)))
    assert upstream_requests == []


async def test_backend_applies_outbound_request_size_even_when_called_directly_async() -> None:
    upstream_requests: list[httpx.Request] = []

    def provider(request: httpx.Request) -> httpx.Response:
        upstream_requests.append(request)
        return httpx.Response(200)

    limits = GatewayLimits(max_request_bytes=80)
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        backend = _backend(client=client, limits=limits)
        with pytest.raises(ModelBackendError) as caught:
            await backend.create_response_async(request=_request(input_items="OFFLINE" * 30))
    assert caught.value.code is ModelBackendErrorCode.UPSTREAM_STREAM_ERROR
    assert upstream_requests == []


async def test_nonstream_backend_uses_only_pinned_host_request_and_returns_original_bytes_async() -> None:
    upstream_requests: list[httpx.Request] = []
    original = _response()

    def provider(request: httpx.Request) -> httpx.Response:
        upstream_requests.append(request)
        return httpx.Response(200, content=original, headers={"Content-Type": "application/json"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(provider), follow_redirects=True, headers={"Authorization": "Bearer wrong"}
    ) as client:
        backend = _backend(client=client)
        result = await backend.create_response_async(
            request=_request(input_items="https://untrusted.invalid/guest-url")
        )

    assert result == original
    assert len(upstream_requests) == 1
    request = upstream_requests[0]
    assert str(request.url) == ENDPOINT
    assert request.method == "POST"
    assert request.headers["authorization"] == f"Bearer {HOST_TOKEN}"
    assert request.headers["accept-encoding"] == "identity"
    assert request.headers["content-type"] == "application/json"
    assert ROUTE.guest_token not in str(request.headers)
    assert "X-PyRIT-Run-ID" not in request.headers
    assert json.loads(request.content)["input"] == "https://untrusted.invalid/guest-url"
    assert json.loads(request.content)["max_output_tokens"] == 16
    assert HOST_TOKEN.encode() not in request.content


async def test_client_default_auth_cannot_override_pinned_host_bearer_async() -> None:
    requests: list[httpx.Request] = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=_response(), headers={"Content-Type": "application/json"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(provider), auth=httpx.BasicAuth("different-user", "different-password")
    ) as client:
        result = await _backend(client=client).create_response_async(request=_request())
    assert result == _response()
    assert len(requests) == 1
    assert requests[0].headers["Authorization"] == f"Bearer {HOST_TOKEN}"


async def test_gateway_observes_exact_nonstream_provider_bytes_without_host_auth_async() -> None:
    original = _response()
    observer = AsyncMock()

    def provider(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=original, headers={"Content-Type": "application/json"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as host_client:
        app = create_codex_responses_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/responses",
                json={"model": ROUTE.model, "input": "OFFLINE", "max_output_tokens": 16},
                headers=GUEST_HEADERS,
            )
    assert reply.status_code == 200
    assert reply.content == original
    records = [call.args[0] for call in observer.await_args_list]
    assert [item.kind for item in records] == [GatewayFrameKind.REQUEST, GatewayFrameKind.RESPONSE]
    assert records[1].frame == original
    assert GatewayCoverage.COMPLETED in records[1].coverage
    assert all(HOST_TOKEN.encode() not in item.frame for item in records)


async def test_stream_splits_arbitrary_chunks_into_exact_sse_frames_and_closes_async() -> None:
    calls = [{**CALL, "id": "fc-1"}]
    initial = {"id": "resp-offline", "object": "response", "model": ROUTE.model, "status": "in_progress"}
    frames = [
        _event(name="response.created", sequence=0, response=initial),
        _event(name="response.output_item.added", sequence=1, item=calls[0]),
        _event(name="response.function_call_arguments.delta", sequence=2, delta='{"command":'),
        _event(name="response.output_item.done", sequence=3, item=calls[0]),
        _event(name="response.completed", sequence=4, response=json.loads(_response(output=calls))),
        b"data: [DONE]\n\n",
    ]
    combined = b"".join(frames)
    chunks = [combined[:1], combined[1:29], combined[29:66], combined[66:80], combined[80:]]
    stream = _ChunkStream(chunks=chunks)

    def provider(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == ENDPOINT
        assert request.headers["authorization"] == f"Bearer {HOST_TOKEN}"
        assert request.headers["accept"] == "text/event-stream"
        assert json.loads(request.content)["tools"] == [FUNCTION]
        return httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream; charset=utf-8"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        backend = _backend(client=client)
        request = _request(streaming=True)
        request.body["tools"] = [FUNCTION]
        actual = [frame async for frame in backend.stream_response_async(request=request)]
    assert actual == frames
    assert stream.closed


async def test_crlf_frames_remain_byte_identical_and_unmodified_async() -> None:
    frames = [frame.replace(b"\n", b"\r\n") for frame in _frames()]
    stream = _ChunkStream(chunks=[b"".join(frames)])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        )
    ) as client:
        backend = _backend(client=client)
        actual = [frame async for frame in backend.stream_response_async(request=_request(streaming=True))]
    assert actual == frames
    assert stream.closed


async def test_gateway_forwards_function_call_and_exact_cli_result_to_model_async() -> None:
    frame_list = _frames(output=[CALL])
    stream = _ChunkStream(chunks=[b"".join(frame_list)])
    seen: list[dict[str, Any]] = []
    observer = AsyncMock()

    def provider(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if body.get("stream"):
            return httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(200, content=_response(), headers={"Content-Type": "application/json"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as host_client:
        app = create_codex_responses_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            first = await guest.post(
                "/v1/responses",
                json={
                    "model": ROUTE.model,
                    "input": "OFFLINE",
                    "stream": True,
                    "tools": [FUNCTION],
                    "max_output_tokens": 16,
                },
                headers=GUEST_HEADERS,
            )
            second = await guest.post(
                "/v1/responses",
                json={
                    "model": ROUTE.model,
                    "input": [
                        CALL,
                        {"type": "function_call_output", "call_id": "call-1", "output": "OFFLINE\nresult"},
                    ],
                    "tools": [FUNCTION],
                    "max_output_tokens": 16,
                },
                headers=GUEST_HEADERS,
            )
    assert first.status_code == second.status_code == 200
    assert first.content == b"".join(frame_list)
    assert seen[1]["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "OFFLINE\nresult",
    }
    events = [
        call.args[0].frame for call in observer.await_args_list if call.args[0].kind == GatewayFrameKind.RESPONSE_EVENT
    ]
    assert events == frame_list
    assert stream.closed


async def test_mixed_reasoning_text_and_custom_tool_sse_keeps_original_usage_and_frames_async() -> None:
    reasoning = {
        "type": "reasoning",
        "id": "reasoning-offline",
        "summary": [{"type": "summary_text", "text": "OFFLINE reasoning"}],
        "encrypted_content": "opaque-offline-content",
    }
    message = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OFFLINE text"}]}
    custom_call = {"type": "custom_tool_call", "name": "apply_patch", "call_id": "patch-1", "input": "OFFLINE patch"}
    in_progress = {"id": "resp-offline", "object": "response", "model": ROUTE.model, "status": "in_progress"}
    frames = [
        _event(name="response.created", sequence=0, response=in_progress),
        _event(name="response.reasoning_summary_text.delta", sequence=1, delta="OFFLINE reasoning"),
        _event(name="response.output_item.added", sequence=2, item={"type": "message", "role": "assistant"}),
        _event(name="response.content_part.added", sequence=3, part={"type": "output_text", "text": ""}),
        _event(name="response.output_text.delta", sequence=4, delta="OFFLINE text"),
        _event(name="response.output_item.done", sequence=5, item=custom_call),
        _event(
            name="response.completed",
            sequence=6,
            response=json.loads(_response(output=[reasoning, message, custom_call])),
        ),
        b"data: [DONE]\n\n",
    ]
    stream = _ChunkStream(chunks=[b"".join(frames)])
    observer = AsyncMock()
    upstream_requests: list[httpx.Request] = []

    def provider(request: httpx.Request) -> httpx.Response:
        upstream_requests.append(request)
        return httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as host_client:
        app = create_codex_responses_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/responses",
                json={
                    "model": ROUTE.model,
                    "input": "OFFLINE",
                    "stream": True,
                    "tools": [CUSTOM_TOOL],
                    "reasoning": {"effort": "low"},
                    "include": ["reasoning.encrypted_content"],
                    "max_output_tokens": 16,
                },
                headers=GUEST_HEADERS,
            )
    assert reply.status_code == 200
    assert reply.content == b"".join(frames)
    assert len(upstream_requests) == 1
    assert json.loads(upstream_requests[0].content)["include"] == ["reasoning.encrypted_content"]
    assert json.loads(upstream_requests[0].content)["tools"] == [CUSTOM_TOOL]
    observed = [call.args[0] for call in observer.await_args_list]
    assert [item.frame for item in observed if item.kind == GatewayFrameKind.RESPONSE_EVENT] == frames
    assert GatewayCoverage.REASONING in observed[-2].coverage
    assert GatewayCoverage.CUSTOM_CALL in observed[-2].coverage
    assert GatewayCoverage.COMPLETED in observed[-2].coverage
    assert stream.closed


@pytest.mark.parametrize("status", [302, 401, 429, 500])
async def test_upstream_status_and_provider_body_never_reach_guest_or_observer_async(status: int) -> None:
    observed: list[httpx.Request] = []
    observer = AsyncMock()

    def provider(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            status,
            content=f"private provider error {HOST_TOKEN}".encode(),
            headers={"Location": "https://untrusted.invalid/redirect", "Content-Type": "text/plain"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider), follow_redirects=True) as host_client:
        app = create_codex_responses_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/responses",
                json={"model": ROUTE.model, "input": "OFFLINE", "max_output_tokens": 16},
                headers=GUEST_HEADERS,
            )
    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == ModelBackendErrorCode.UPSTREAM_HTTP_ERROR.value
    assert len(observed) == 1
    assert str(observed[0].url) == ENDPOINT
    records = [call.args[0] for call in observer.await_args_list]
    assert records[-1].kind == GatewayFrameKind.GATEWAY_ERROR
    assert records[-1].error_code == "upstream_http_error"
    assert records[-1].status_code == 502
    assert GatewayCoverage.FAILED in records[-1].coverage
    assert records[-1].frame == reply.content
    assert all(HOST_TOKEN.encode() not in item.frame for item in records)
    assert b"private provider error" not in reply.content
    assert b"untrusted.invalid" not in reply.content


async def test_upstream_connect_error_is_sanitized_and_observed_async() -> None:
    observer = AsyncMock()

    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"private url/token {HOST_TOKEN}", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as host_client:
        app = create_codex_responses_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/responses",
                json={"model": ROUTE.model, "input": "OFFLINE", "max_output_tokens": 16},
                headers=GUEST_HEADERS,
            )
    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == "upstream_network_error"
    assert observer.await_args_list[-1].args[0].error_code == "upstream_network_error"
    assert HOST_TOKEN not in reply.text
    assert all(HOST_TOKEN.encode() not in call.args[0].frame for call in observer.await_args_list)


async def test_slow_upstream_dispatch_returns_typed_timeout_and_observation_async() -> None:
    observer = AsyncMock()
    limits = GatewayLimits(max_request_bytes=4_096, max_response_bytes=8_192, timeout_seconds=1.0)

    async def slow_provider(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, content=_response(), headers={"Content-Type": "application/json"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(slow_provider)) as host_client:
        app = create_codex_responses_app(
            route=ROUTE,
            limits=limits,
            backend=_backend(client=host_client, limits=limits, timeout_seconds=0.05),
            observation_callback=observer,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/responses",
                json={"model": ROUTE.model, "input": "OFFLINE", "max_output_tokens": 16},
                headers=GUEST_HEADERS,
            )
    assert reply.status_code == 504
    assert reply.json()["error"]["code"] == "upstream_timeout"
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert observer.await_args_list[-1].args[0].status_code == 504


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        ({"Content-Type": "text/plain"}, b"OFFLINE"),
        ({"Content-Type": "application/json", "Content-Encoding": "gzip"}, b"OFFLINE"),
        ({"Content-Type": "application/json"}, b"{bad-json}"),
        ({"Content-Type": "application/json"}, b'{"output":[{"text":"host-only-' + b"y" * 32 + b'"}]}'),
    ],
)
async def test_bad_upstream_body_is_rejected_before_observation_async(headers: dict[str, str], body: bytes) -> None:
    observer = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body, headers=headers))
    ) as host_client:
        app = create_codex_responses_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/responses",
                json={"model": ROUTE.model, "input": "OFFLINE", "max_output_tokens": 16},
                headers=GUEST_HEADERS,
            )
    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == "upstream_stream_error"
    assert HOST_TOKEN not in reply.text
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert all(HOST_TOKEN.encode() not in call.args[0].frame for call in observer.await_args_list)


@pytest.mark.parametrize("advertised_size", [None, 10_000])
async def test_upstream_response_byte_limit_rejects_header_or_actual_body_async(advertised_size: int | None) -> None:
    raw = _response()
    limits = GatewayLimits(max_request_bytes=4_096, max_response_bytes=len(raw) - 1)
    stream = _ChunkStream(chunks=[raw[:12], raw[12:]])
    headers = {"Content-Type": "application/json"}
    if advertised_size is not None:
        headers["Content-Length"] = str(advertised_size)
    else:
        headers["Content-Length"] = "1"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream, headers=headers))
    ) as host_client:
        backend = _backend(client=host_client, limits=limits)
        with pytest.raises(ModelBackendError) as caught:
            await backend.create_response_async(request=_request())
    assert caught.value.code is ModelBackendErrorCode.UPSTREAM_STREAM_ERROR
    assert stream.closed


async def test_streamed_auth_is_held_across_events_and_never_forwarded_async() -> None:
    created = _frames()[0]
    first_half = HOST_TOKEN[: len(HOST_TOKEN) // 2]
    second_half = HOST_TOKEN[len(HOST_TOKEN) // 2 :]
    first = _event(name="response.output_text.delta", sequence=1, delta=first_half)
    second = _event(name="response.output_text.delta", sequence=2, delta=second_half)
    stream = _ChunkStream(chunks=[created + first[:13], first[13:] + second])
    observer = AsyncMock()

    def provider(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as host_client:
        app = create_codex_responses_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/responses",
                json={
                    "model": ROUTE.model,
                    "input": "OFFLINE",
                    "tools": [FUNCTION],
                    "stream": True,
                    "max_output_tokens": 16,
                },
                headers=GUEST_HEADERS,
            )
    assert reply.status_code == 200
    assert reply.content.startswith(created)
    assert b"upstream_stream_error" in reply.content
    assert first_half.encode() not in reply.content
    assert second_half.encode() not in reply.content
    assert b"data: [DONE]" not in reply.content
    assert stream.closed
    assert observer.await_args_list[-1].args[0].error_code == "upstream_stream_error"
    assert all(first_half.encode() not in call.args[0].frame for call in observer.await_args_list)


async def test_json_escaped_host_auth_is_rejected_before_stream_observation_async() -> None:
    created = _frames()[0]
    escaped = HOST_TOKEN.replace("h", "\\u0068", 1)
    delta = (
        b"event: response.output_text.delta\n"
        + f'data: {{"type":"response.output_text.delta","sequence_number":1,"delta":"{escaped}"}}\n\n'.encode()
    )
    stream = _ChunkStream(chunks=[created, delta])
    observer = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        )
    ) as host_client:
        app = create_codex_responses_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/responses",
                json={
                    "model": ROUTE.model,
                    "input": "OFFLINE",
                    "tools": [FUNCTION],
                    "stream": True,
                    "max_output_tokens": 16,
                },
                headers=GUEST_HEADERS,
            )
    assert reply.status_code == 200
    assert b"upstream_stream_error" in reply.content
    assert HOST_TOKEN.encode() not in reply.content
    assert delta not in reply.content
    assert all(delta not in call.args[0].frame for call in observer.await_args_list)
    assert stream.closed


@pytest.mark.parametrize("mode", ["truncated", "network", "timeout"])
async def test_partial_stream_failure_is_typed_observed_and_closes_async(mode: str) -> None:
    created = _frames()[0]
    failure = httpx.ReadError(f"private host error {HOST_TOKEN}") if mode == "network" else None
    stream = _ChunkStream(
        chunks=[created],
        failure=failure,
        delay_seconds=0.1 if mode == "timeout" else 0,
    )
    observer = AsyncMock()
    backend_timeout = 0.03 if mode == "timeout" else None

    def provider(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as host_client:
        app = create_codex_responses_app(
            route=ROUTE,
            limits=GatewayLimits(max_request_bytes=4_096, max_response_bytes=8_192, timeout_seconds=1.0),
            backend=_backend(
                client=host_client,
                limits=GatewayLimits(max_request_bytes=4_096, max_response_bytes=8_192, timeout_seconds=1.0),
                timeout_seconds=backend_timeout,
            ),
            observation_callback=observer,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/responses",
                json={
                    "model": ROUTE.model,
                    "input": "OFFLINE",
                    "tools": [FUNCTION],
                    "stream": True,
                    "max_output_tokens": 16,
                },
                headers=GUEST_HEADERS,
            )
    expected = (
        "upstream_network_error"
        if mode == "network"
        else ("upstream_timeout" if mode == "timeout" else "upstream_stream_error")
    )
    assert reply.status_code == 200
    assert reply.content.startswith(created)
    assert expected.encode() in reply.content
    assert b"data: [DONE]" not in reply.content
    assert HOST_TOKEN.encode() not in reply.content
    assert stream.closed
    boundary = observer.await_args_list[-1].args[0]
    assert boundary.kind == GatewayFrameKind.GATEWAY_ERROR
    assert boundary.error_code == expected
    assert boundary.frame in reply.content


async def test_first_frame_failure_is_json_error_with_observed_boundary_async() -> None:
    stream = _ChunkStream(chunks=[b"not an SSE frame\n\n"])
    observer = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        )
    ) as host_client:
        app = create_codex_responses_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/responses",
                json={
                    "model": ROUTE.model,
                    "input": "OFFLINE",
                    "tools": [FUNCTION],
                    "stream": True,
                    "max_output_tokens": 16,
                },
                headers=GUEST_HEADERS,
            )
    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == "upstream_stream_error"
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert observer.await_args_list[-1].args[0].coverage == frozenset(
        {GatewayCoverage.STREAMING, GatewayCoverage.FAILED}
    )
    assert observer.await_args_list[-1].args[0].frame == reply.content
    assert stream.closed


async def test_observer_failure_preserves_primary_typed_backend_error_async(caplog: pytest.LogCaptureFixture) -> None:
    observer = AsyncMock(side_effect=[None, RuntimeError(f"private recorder {HOST_TOKEN}")])

    def provider(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=f"provider {HOST_TOKEN}", headers={"Content-Type": "text/plain"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as host_client:
        app = create_codex_responses_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/responses",
                json={"model": ROUTE.model, "input": "OFFLINE", "max_output_tokens": 16},
                headers=GUEST_HEADERS,
            )
    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == "upstream_http_error"
    assert "upstream_http_error" in caplog.text
    assert HOST_TOKEN not in reply.text + caplog.text
    assert observer.await_args_list[-1].args[0].error_code == "upstream_http_error"


async def test_partial_stream_observer_failure_keeps_typed_primary_error_async(
    caplog: pytest.LogCaptureFixture,
) -> None:
    created = _frames()[0]
    stream = _ChunkStream(chunks=[created], failure=httpx.ReadError(f"private {HOST_TOKEN}"))
    observer = AsyncMock(side_effect=[None, None, RuntimeError(f"recorder {HOST_TOKEN}")])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        )
    ) as host_client:
        app = create_codex_responses_app(
            route=ROUTE, limits=LIMITS, backend=_backend(client=host_client), observation_callback=observer
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            reply = await guest.post(
                "/v1/responses",
                json={
                    "model": ROUTE.model,
                    "input": "OFFLINE",
                    "tools": [FUNCTION],
                    "stream": True,
                    "max_output_tokens": 16,
                },
                headers=GUEST_HEADERS,
            )
    assert reply.status_code == 200
    assert reply.content.startswith(created)
    assert b"upstream_network_error" in reply.content
    assert b"observation_failed" not in reply.content
    assert b"data: [DONE]" not in reply.content
    assert HOST_TOKEN not in reply.text + caplog.text
    assert observer.await_args_list[-1].args[0].error_code == "upstream_network_error"
    assert stream.closed


async def test_cancelling_partial_stream_closes_upstream_without_done_async() -> None:
    stream = _ChunkStream(chunks=[_frames()[0]], wait_forever=True)

    def provider(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as host_client:
        app = create_codex_responses_app(route=ROUTE, limits=LIMITS, backend=_backend(client=host_client))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid"
        ) as guest:
            task = asyncio.create_task(
                guest.post(
                    "/v1/responses",
                    json={
                        "model": ROUTE.model,
                        "input": "OFFLINE",
                        "tools": [FUNCTION],
                        "stream": True,
                        "max_output_tokens": 16,
                    },
                    headers=GUEST_HEADERS,
                )
            )
            try:
                await asyncio.wait_for(stream.waiting.wait(), timeout=2)
            finally:
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert stream.closed
