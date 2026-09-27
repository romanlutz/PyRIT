# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from pyrit.prompt_target.gateway.codex_responses import create_codex_responses_app
from pyrit.prompt_target.gateway.responses_contract import (
    BackendCapabilities,
    GatewayCoverage,
    GatewayFrameKind,
    GatewayLimits,
    GatewayObservation,
    GatewayRoute,
    ModelRequest,
)

ROUTE = GatewayRoute(run_id="run-offline-1", model="codex-fixture", guest_token="sandbox-only-" + "x" * 32)
HEADERS = {"Authorization": f"Bearer {ROUTE.guest_token}", "X-PyRIT-Run-ID": ROUTE.run_id}
FUNCTION = {
    "type": "function",
    "name": "shell_command",
    "description": "Run a command in the CLI's sandbox",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
}
CUSTOM_TOOL = {"type": "custom", "name": "apply_patch", "format": {"type": "text"}}
CALL = {
    "type": "function_call",
    "id": "fc-1",
    "call_id": "call-1",
    "name": "shell_command",
    "arguments": '{"command":"echo OFFLINE"}',
}
TEXT = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OFFLINE answer"}]}


def _response(
    *,
    output: list[dict[str, Any]] | None = None,
    status: str = "completed",
    model: str = ROUTE.model,
    tokens: int = 3,
) -> dict[str, Any]:
    return {
        "id": "resp-offline",
        "object": "response",
        "model": model,
        "status": status,
        "output": output if output is not None else [TEXT],
        "usage": {"input_tokens": 7, "output_tokens": tokens},
    }


def _wire(*, output: list[dict[str, Any]] | None = None, tokens: int = 3) -> bytes:
    return json.dumps(_response(output=output, tokens=tokens), separators=(",", ":")).encode()


def _event(*, name: str, sequence: int, **details: Any) -> bytes:
    payload = {"type": name, "sequence_number": sequence, **details}
    return f"event: {name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


class FakeModelOnlyBackend:
    capabilities = BackendCapabilities(streaming=True, function_tools=True, custom_tools=True, reasoning=True)

    def __init__(
        self,
        *,
        responses: list[bytes] | None = None,
        streams: list[list[bytes]] | None = None,
    ) -> None:
        self.responses = responses if responses is not None else [_wire()]
        self.streams = streams if streams is not None else []
        self.requests: list[ModelRequest] = []
        self.closed = False

    async def create_response_async(self, *, request: ModelRequest) -> bytes:
        self.requests.append(request)
        return self.responses.pop(0)

    async def stream_response_async(self, *, request: ModelRequest) -> AsyncGenerator[bytes, None]:
        self.requests.append(request)
        try:
            for frame in self.streams.pop(0):
                yield frame
        finally:
            self.closed = True


def _client(
    *,
    backend: FakeModelOnlyBackend | None,
    limits: GatewayLimits | None = None,
    observer: AsyncMock | None = None,
) -> httpx.AsyncClient:
    app = create_codex_responses_app(
        route=ROUTE,
        limits=limits or GatewayLimits(),
        backend=backend,
        observation_callback=observer,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid")


async def test_nonstream_preserves_original_bytes_and_uses_only_host_routing_async() -> None:
    original = b'{"model":"codex-fixture","input":"OFFLINE prompt","store":false}'
    returned = _wire()
    backend = FakeModelOnlyBackend(responses=[returned])
    observe = AsyncMock()
    async with _client(backend=backend, observer=observe) as client:
        response = await client.post(
            "/v1/responses", content=original, headers={**HEADERS, "Content-Type": "application/json"}
        )

    assert response.status_code == 200
    assert response.content == returned
    assert response.headers["cache-control"] == "no-store"
    sent = backend.requests[0]
    assert sent.run_id == ROUTE.run_id
    assert sent.body == {
        "model": ROUTE.model,
        "input": "OFFLINE prompt",
        "store": False,
        "max_output_tokens": GatewayLimits().max_output_tokens_per_request,
    }
    assert not hasattr(sent, "headers")
    assert ROUTE.guest_token.encode() not in json.dumps(sent.body).encode()
    assert observe.await_count == 2
    first, second = (call.args[0] for call in observe.await_args_list)
    assert isinstance(first, GatewayObservation)
    assert [first.kind, second.kind] == [GatewayFrameKind.REQUEST, GatewayFrameKind.RESPONSE]
    assert [first.frame, second.frame] == [original, returned]
    assert first.request_id == second.request_id == sent.request_id
    assert second.coverage == frozenset({GatewayCoverage.COMPLETED})


async def test_streaming_tool_call_and_cli_owned_result_are_forwarded_unchanged_async() -> None:
    created = _response(output=[], status="in_progress")
    frames = [
        _event(name="response.created", sequence=0, response=created),
        _event(name="response.output_item.added", sequence=1, item=CALL),
        _event(
            name="response.function_call_arguments.delta",
            sequence=2,
            item_id=CALL["id"],
            delta='{"command":',
        ),
        _event(name="response.output_item.done", sequence=3, item=CALL),
        _event(name="response.completed", sequence=4, response=_response(output=[CALL])),
        b"data: [DONE]\n\n",
    ]
    backend = FakeModelOnlyBackend(streams=[frames], responses=[_wire()])
    observe = AsyncMock()
    request = {
        "model": ROUTE.model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "OFFLINE prompt"}]}],
        "tools": [FUNCTION],
        "parallel_tool_calls": True,
        "stream": True,
        "store": False,
    }
    async with _client(backend=backend, observer=observe) as client:
        reply = await client.post("/v1/responses", json=request, headers=HEADERS)
        follow_up = await client.post(
            "/v1/responses",
            json={
                "model": ROUTE.model,
                "input": [
                    {"role": "user", "content": "OFFLINE prompt"},
                    CALL,
                    {"type": "function_call_output", "call_id": CALL["call_id"], "output": "OFFLINE\nresult"},
                ],
                "tools": [FUNCTION],
            },
            headers=HEADERS,
        )
    assert reply.status_code == follow_up.status_code == 200
    assert reply.content == b"".join(frames)
    assert backend.closed
    assert len(backend.requests) == 2
    assert backend.requests[0].body["tools"] == [FUNCTION]
    assert backend.requests[1].body["input"][-1]["output"] == "OFFLINE\nresult"
    observations = [call.args[0] for call in observe.await_args_list]
    assert [entry.frame for entry in observations[1:7]] == frames
    assert observations[0].coverage >= frozenset({GatewayCoverage.FUNCTION_TOOL, GatewayCoverage.STREAMING})
    assert GatewayCoverage.FUNCTION_CALL in observations[4].coverage
    assert GatewayCoverage.FUNCTION_RESULT in observations[7].coverage


async def test_custom_text_tool_call_and_result_remain_cli_owned_async() -> None:
    call = {
        "type": "custom_tool_call",
        "name": "apply_patch",
        "call_id": "call-patch",
        "input": "*** Begin Patch\n*** End Patch\n",
    }
    backend = FakeModelOnlyBackend(responses=[_wire(output=[call]), _wire()])
    async with _client(backend=backend) as client:
        result = await client.post(
            "/v1/responses",
            json={"model": ROUTE.model, "input": "OFFLINE", "tools": [CUSTOM_TOOL]},
            headers=HEADERS,
        )
        await client.post(
            "/v1/responses",
            json={
                "model": ROUTE.model,
                "input": [call, {"type": "custom_tool_call_output", "call_id": "call-patch", "output": "OFFLINE"}],
                "tools": [CUSTOM_TOOL],
            },
            headers=HEADERS,
        )
    assert result.status_code == 200
    assert result.json()["output"] == [call]
    assert backend.requests[1].body["input"][1]["call_id"] == "call-patch"


async def test_custom_tool_stream_preserves_input_deltas_and_terminal_call_async() -> None:
    call = {"type": "custom_tool_call", "name": "apply_patch", "call_id": "patch-1", "input": "OFFLINE patch"}
    frames = [
        _event(name="response.created", sequence=0, response=_response(output=[], status="in_progress")),
        _event(name="response.output_item.added", sequence=1, item={**call, "input": ""}),
        _event(name="response.custom_tool_call_input.delta", sequence=2, delta="OFFLINE patch", item_id="ct-1"),
        _event(name="response.output_item.done", sequence=3, item=call),
        _event(name="response.completed", sequence=4, response=_response(output=[call])),
        b"data: [DONE]\n\n",
    ]
    backend = FakeModelOnlyBackend(streams=[frames])
    async with _client(backend=backend) as client:
        reply = await client.post(
            "/v1/responses",
            json={"model": ROUTE.model, "input": "OFFLINE", "tools": [CUSTOM_TOOL], "stream": True},
            headers=HEADERS,
        )
    assert reply.status_code == 200
    assert reply.content == b"".join(frames)
    assert backend.closed


@pytest.mark.parametrize(
    ("headers", "status", "code"),
    [
        ({}, 401, "invalid_token"),
        ({"Authorization": "Bearer wrong", "X-PyRIT-Run-ID": ROUTE.run_id}, 401, "invalid_token"),
        ({"Authorization": "Bearer " + ROUTE.guest_token}, 403, "invalid_run"),
        ({"Authorization": "Bearer " + ROUTE.guest_token, "X-PyRIT-Run-ID": "other-run"}, 403, "invalid_run"),
        (
            [("Authorization", f"Bearer {ROUTE.guest_token}"), ("Authorization", f"Bearer {ROUTE.guest_token}")],
            401,
            "invalid_token",
        ),
        (
            [
                ("Authorization", f"Bearer {ROUTE.guest_token}"),
                ("X-PyRIT-Run-ID", ROUTE.run_id),
                ("X-PyRIT-Run-ID", ROUTE.run_id),
            ],
            403,
            "invalid_run",
        ),
        (
            [(b"Authorization", b"Bearer \xff"), (b"X-PyRIT-Run-ID", ROUTE.run_id.encode())],
            401,
            "invalid_token",
        ),
    ],
)
async def test_authentication_rejects_unrouted_requests_before_observation_async(
    headers: dict[str, str] | list[tuple[str, str]], status: int, code: str
) -> None:
    backend = FakeModelOnlyBackend()
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE"}, headers=headers)
    assert reply.status_code == status
    assert reply.json()["error"]["code"] == code
    assert ROUTE.guest_token not in reply.text
    assert backend.requests == []
    observer.assert_not_awaited()


async def test_only_exact_responses_path_and_post_are_mounted_async() -> None:
    backend = FakeModelOnlyBackend()
    async with _client(backend=backend) as client:
        for path in ("/v1/models", "/v1/chat/completions", "/v1/responses/"):
            assert (await client.post(path, json={"model": ROUTE.model}, headers=HEADERS)).status_code == 404
        assert (await client.get("/v1/responses", headers=HEADERS)).status_code == 405
        query = await client.post(
            "/v1/responses?url=https://untrusted.invalid",
            json={"model": ROUTE.model, "input": "OFFLINE"},
            headers=HEADERS,
        )
        assert query.status_code == 501
        assert query.json()["error"]["code"] == "unsupported_feature"
    assert backend.requests == []


@pytest.mark.parametrize(
    ("extra", "status", "code"),
    [
        ({"model": "unrouted-model"}, 400, "invalid_request"),
        ({"previous_response_id": "resp-saved"}, 501, "unsupported_feature"),
        ({"background": True}, 501, "unsupported_feature"),
        ({"store": True}, 501, "unsupported_feature"),
        ({"tools": [{"type": "web_search_preview"}]}, 501, "unsupported_feature"),
        (
            {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://bad.invalid"}]}]},
            501,
            "unsupported_feature",
        ),
        ({"text": {"format": {"type": "json_schema", "schema": {}}}}, 501, "unsupported_feature"),
        (
            {"tools": [{"type": "custom", "name": "curl", "format": {"type": "grammar", "definition": "..."}}]},
            501,
            "unsupported_feature",
        ),
        ({"max_output_tokens": True}, 400, "invalid_request"),
        ({"max_output_tokens": 9_000}, 429, "output_token_limit"),
    ],
)
async def test_unsupported_and_over_budget_requests_never_reach_backend_async(
    extra: dict[str, Any], status: int, code: str
) -> None:
    backend = FakeModelOnlyBackend()
    observe = AsyncMock()
    async with _client(backend=backend, observer=observe) as client:
        reply = await client.post(
            "/v1/responses",
            json={"model": ROUTE.model, "input": "OFFLINE", **extra},
            headers=HEADERS,
        )
    assert reply.status_code == status
    assert reply.json()["error"]["code"] == code
    assert backend.requests == []
    observe.assert_not_awaited()


async def test_no_backend_fails_explicitly_rather_than_reusing_target_tool_loop_async() -> None:
    async with _client(backend=None) as client:
        reply = await client.post("/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE"}, headers=HEADERS)
    assert reply.status_code == 501
    assert reply.json()["error"]["code"] == "model_backend_required"
    assert "OpenAIResponseTarget" in reply.json()["error"]["message"]


@pytest.mark.parametrize(
    "options",
    [
        {"stream": True},
        {"tools": [FUNCTION]},
        {"tools": [CUSTOM_TOOL]},
        {"reasoning": {"effort": "low"}},
        {"include": ["reasoning.encrypted_content"]},
    ],
)
async def test_backend_cannot_claim_capabilities_it_does_not_implement_async(options: dict[str, Any]) -> None:
    backend = FakeModelOnlyBackend()
    backend.capabilities = BackendCapabilities()
    async with _client(backend=backend) as client:
        reply = await client.post(
            "/v1/responses",
            json={"model": ROUTE.model, "input": "OFFLINE", **options},
            headers=HEADERS,
        )
    assert reply.status_code == 501
    assert reply.json()["error"]["code"] == "unsupported_feature"
    assert backend.requests == []


@pytest.mark.parametrize(
    ("extra", "status"),
    [
        ({"input": [{"type": []}]}, 400),
        ({"input": [{"role": [], "content": "OFFLINE"}]}, 400),
        ({"input": [{"role": "user", "content": [{"type": [], "text": "OFFLINE"}]}]}, 400),
        (
            {
                "input": [
                    {"type": "function_call", "call_id": "1", "name": "shell_command", "arguments": "{}", "status": []}
                ]
            },
            400,
        ),
        ({"tools": [{"type": []}]}, 400),
        ({"reasoning": {"effort": []}}, 400),
        ({"text": {"verbosity": []}}, 400),
        ({"tool_choice": {"type": [], "name": "shell_command"}}, 501),
        ({"input": [{"role": "user", "content": [{"type": "output_text", "text": "x", "annotations": [{}]}]}]}, 501),
    ],
)
async def test_malformed_nested_guest_fields_fail_explicitly_async(extra: dict[str, Any], status: int) -> None:
    backend = FakeModelOnlyBackend()
    async with _client(backend=backend) as client:
        reply = await client.post(
            "/v1/responses",
            json={"model": ROUTE.model, "input": "OFFLINE", **extra},
            headers=HEADERS,
        )
    assert reply.status_code == status
    assert backend.requests == []


@pytest.mark.parametrize(
    ("content", "headers", "limits", "status", "code"),
    [
        (b"{not-json}", {**HEADERS, "Content-Type": "application/json"}, None, 400, "invalid_json"),
        (b"[]", {**HEADERS, "Content-Type": "application/json"}, None, 400, "invalid_json"),
        (
            b'{"model":"codex-fixture","input":"OFFLINE"}',
            {**HEADERS, "Content-Type": "application/json", "Content-Length": "9" * 100},
            None,
            400,
            "invalid_content_length",
        ),
        (
            b'{"model":"codex-fixture","input":"OFFLINE"}',
            {**HEADERS, "Content-Type": "text/plain"},
            None,
            415,
            "invalid_content_type",
        ),
        (
            b'{"model":"codex-fixture","input":"OFFLINE"}',
            {**HEADERS, "Content-Type": "application/json"},
            GatewayLimits(max_request_bytes=10),
            413,
            "request_too_large",
        ),
    ],
)
async def test_invalid_or_oversized_bodies_are_rejected_async(
    content: bytes, headers: dict[str, str], limits: GatewayLimits | None, status: int, code: str
) -> None:
    backend = FakeModelOnlyBackend()
    async with _client(backend=backend, limits=limits) as client:
        reply = await client.post("/v1/responses", content=content, headers=headers)
    assert reply.status_code == status
    assert reply.json()["error"]["code"] == code
    assert backend.requests == []


@pytest.mark.parametrize(
    "raw",
    [
        b'{"model":"codex-fixture","model":"codex-fixture","input":"OFFLINE"}',
        b'{"model":"codex-fixture","input":"OFFLINE","temperature":NaN}',
        b'{"model":"codex-fixture","input":"OFFLINE","temperature":1e999}',
        b"[" * 1_200 + b"0" + b"]" * 1_200,
    ],
)
async def test_nonstandard_or_nested_json_is_rejected_before_backend_async(raw: bytes) -> None:
    backend = FakeModelOnlyBackend()
    async with _client(backend=backend) as client:
        reply = await client.post("/v1/responses", content=raw, headers={**HEADERS, "Content-Type": "application/json"})
    assert reply.status_code == 400
    assert reply.json()["error"]["code"] == "invalid_json"
    assert backend.requests == []


async def test_both_request_and_reserved_token_budgets_are_atomic_async() -> None:
    backend = FakeModelOnlyBackend(responses=[_wire(tokens=3), _wire(tokens=3)])
    limits = GatewayLimits(max_requests=2, max_output_tokens_per_request=4, max_total_output_tokens=6)
    async with _client(backend=backend, limits=limits) as client:
        first = await client.post(
            "/v1/responses",
            json={"model": ROUTE.model, "input": "OFFLINE", "max_output_tokens": 3},
            headers=HEADERS,
        )
        second = await client.post(
            "/v1/responses",
            json={"model": ROUTE.model, "input": "OFFLINE", "max_output_tokens": 3},
            headers=HEADERS,
        )
        third = await client.post(
            "/v1/responses",
            json={"model": ROUTE.model, "input": "OFFLINE", "max_output_tokens": 3},
            headers=HEADERS,
        )
    assert [first.status_code, second.status_code, third.status_code] == [200, 200, 429]
    assert third.json()["error"]["code"] == "request_budget"
    assert len(backend.requests) == 2

    backend = FakeModelOnlyBackend(responses=[_wire()])
    async with _client(backend=backend, limits=limits) as client:
        assert (
            await client.post(
                "/v1/responses",
                json={"model": ROUTE.model, "input": "OFFLINE", "max_output_tokens": 3},
                headers=HEADERS,
            )
        ).status_code == 200
        exhausted = await client.post("/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE"}, headers=HEADERS)
    assert exhausted.status_code == 429
    assert exhausted.json()["error"]["code"] == "token_budget"
    assert len(backend.requests) == 1


async def test_total_input_byte_budget_cannot_be_reused_after_a_request_async() -> None:
    raw = b'{"model":"codex-fixture","input":"OFFLINE","max_output_tokens":3}'
    limits = GatewayLimits(max_total_request_bytes=2 * len(raw) - 1)
    backend = FakeModelOnlyBackend(responses=[_wire()])
    async with _client(backend=backend, limits=limits) as client:
        first = await client.post("/v1/responses", content=raw, headers={**HEADERS, "Content-Type": "application/json"})
        second = await client.post(
            "/v1/responses", content=raw, headers={**HEADERS, "Content-Type": "application/json"}
        )
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "input_byte_budget"
    assert len(backend.requests) == 1


async def test_concurrent_requests_cannot_exceed_one_reserved_slot_async() -> None:
    backend = FakeModelOnlyBackend(responses=[_wire()])
    async with _client(backend=backend, limits=GatewayLimits(max_requests=1)) as client:
        first, second = await asyncio.gather(
            client.post("/v1/responses", json={"model": ROUTE.model, "input": "A"}, headers=HEADERS),
            client.post("/v1/responses", json={"model": ROUTE.model, "input": "B"}, headers=HEADERS),
        )
    assert sorted([first.status_code, second.status_code]) == [200, 429]
    assert len(backend.requests) == 1


@pytest.mark.parametrize(
    "bad_response",
    [
        _wire(tokens=9_000),
        json.dumps({**_response(), "usage": None}).encode(),
        json.dumps(_response(model="another-model")).encode(),
        json.dumps(_response(output=[{"type": "web_search_call"}])).encode(),
        json.dumps(_response(output=[{"type": "message", "role": "assistant", "content": [[]]}])).encode(),
        json.dumps(_response(output=[{"type": ["web_search_call"]}])).encode(),
        json.dumps(_response(output=[CALL])).encode(),
        json.dumps({**_response(), "error": {"message": "provider failed"}}).encode(),
        b'{"model":"codex-fixture","model":"codex-fixture","object":"response"}',
        json.dumps({**_response(), "usage": {"output_tokens": float("nan")}}).encode(),
        b"not-json",
        b"[]",
    ],
)
async def test_bad_backend_output_fails_without_fabricated_success_async(bad_response: bytes) -> None:
    backend = FakeModelOnlyBackend(responses=[bad_response])
    async with _client(backend=backend) as client:
        reply = await client.post("/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE"}, headers=HEADERS)
    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == "invalid_backend_response"


async def test_nonstream_response_byte_limit_applies_before_forwarding_async() -> None:
    backend = FakeModelOnlyBackend(responses=[_wire()])
    limits = GatewayLimits(max_response_bytes=12)
    async with _client(backend=backend, limits=limits) as client:
        reply = await client.post("/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE"}, headers=HEADERS)
    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == "response_too_large"


async def test_incomplete_response_is_preserved_and_observed_as_incomplete_async() -> None:
    original = json.dumps(_response(status="incomplete", output=[])).encode()
    observer = AsyncMock()
    backend = FakeModelOnlyBackend(responses=[original])
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post("/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE"}, headers=HEADERS)
    assert reply.status_code == 200
    assert reply.content == original
    assert GatewayCoverage.INCOMPLETE in observer.await_args_list[1].args[0].coverage


@pytest.mark.parametrize(
    ("frames", "error_code"),
    [
        (
            [_event(name="response.created", sequence=0, response=_response(output=[], status="in_progress"))],
            "incomplete_stream",
        ),
        (
            [
                _event(name="response.created", sequence=0, response=_response(output=[], status="in_progress")),
                _event(name="response.output_text.delta", sequence=0, delta="duplicate"),
            ],
            "invalid_backend_response",
        ),
        (
            [
                _event(name="response.created", sequence=0, response=_response(output=[], status="in_progress")),
                _event(name="response.output_item.done", sequence=1, item={"type": "web_search_call"}),
            ],
            "invalid_backend_response",
        ),
    ],
)
async def test_bad_streams_emit_error_instead_of_success_async(frames: list[bytes], error_code: str) -> None:
    backend = FakeModelOnlyBackend(streams=[frames])
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post(
            "/v1/responses",
            json={"model": ROUTE.model, "input": "OFFLINE", "stream": True},
            headers=HEADERS,
        )
    assert reply.status_code == 200
    assert reply.content.startswith(frames[0])
    assert b"event: error\n" in reply.content
    assert error_code.encode() in reply.content
    assert b"data: [DONE]" not in reply.content
    assert backend.closed
    boundary = observer.await_args_list[-1].args[0]
    assert boundary.kind == GatewayFrameKind.GATEWAY_ERROR
    assert boundary.frame.startswith(b"event: error\n")
    assert reply.content.endswith(boundary.frame)
    assert boundary.error_code == error_code
    assert boundary.status_code == 502
    assert boundary.coverage == frozenset({GatewayCoverage.STREAMING, GatewayCoverage.FAILED})


async def test_invalid_first_stream_frame_returns_http_error_async() -> None:
    backend = FakeModelOnlyBackend(streams=[[_event(name="response.output_text.delta", sequence=0, delta="hi")]])
    async with _client(backend=backend) as client:
        reply = await client.post(
            "/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE", "stream": True}, headers=HEADERS
        )
    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == "invalid_backend_response"
    assert backend.closed


async def test_stream_byte_ceiling_returns_error_and_closes_provider_async() -> None:
    created = _event(name="response.created", sequence=0, response=_response(output=[], status="in_progress"))
    oversized = _event(name="response.output_text.delta", sequence=1, delta="X" * 100)
    backend = FakeModelOnlyBackend(streams=[[created, oversized]])
    limits = GatewayLimits(max_response_bytes=len(created) + 2)
    observer = AsyncMock()
    async with _client(backend=backend, limits=limits, observer=observer) as client:
        reply = await client.post(
            "/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE", "stream": True}, headers=HEADERS
        )
    assert reply.status_code == 200
    assert reply.content.startswith(created)
    assert b"response_too_large" in reply.content
    assert oversized not in reply.content
    assert backend.closed
    assert observer.await_args_list[-1].args[0].error_code == "response_too_large"


async def test_bad_backend_sse_json_fails_explicitly_after_initial_event_async() -> None:
    created = _event(name="response.created", sequence=0, response=_response(output=[], status="in_progress"))
    bad = (
        b"event: response.output_text.delta\n"
        b'data: {"type":"response.output_text.delta","type":"response.output_text.delta"}\n\n'
    )
    backend = FakeModelOnlyBackend(streams=[[created, bad]])
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post(
            "/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE", "stream": True}, headers=HEADERS
        )
    assert reply.status_code == 200
    assert reply.content.startswith(created)
    assert b"invalid_backend_response" in reply.content
    assert bad not in reply.content
    assert backend.closed
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert observer.await_args_list[-1].args[0].error_code == "invalid_backend_response"


async def test_backend_stream_exception_reports_error_without_leaking_exception_async() -> None:
    class BrokenStreamBackend(FakeModelOnlyBackend):
        async def stream_response_async(self, *, request: ModelRequest) -> AsyncGenerator[bytes, None]:
            self.requests.append(request)
            try:
                yield _event(name="response.created", sequence=0, response=_response(output=[], status="in_progress"))
                raise RuntimeError("private provider detail")
            finally:
                self.closed = True

    backend = BrokenStreamBackend()
    observer = AsyncMock()
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post(
            "/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE", "stream": True}, headers=HEADERS
        )
    assert reply.status_code == 200
    assert b"backend_failed" in reply.content
    assert b"private provider detail" not in reply.content
    assert b"data: [DONE]" not in reply.content
    assert backend.closed
    assert observer.await_args_list[-1].args[0].error_code == "backend_failed"


async def test_model_failed_stream_remains_failed_not_completed_async() -> None:
    failed = {**_response(output=[], status="failed"), "usage": None, "error": {"code": "offline_failure"}}
    frames = [
        _event(name="response.created", sequence=0, response=_response(output=[], status="in_progress")),
        _event(name="response.failed", sequence=1, response=failed),
        b"data: [DONE]\n\n",
    ]
    backend = FakeModelOnlyBackend(streams=[frames])
    observe = AsyncMock()
    async with _client(backend=backend, observer=observe) as client:
        reply = await client.post(
            "/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE", "stream": True}, headers=HEADERS
        )
    assert reply.status_code == 200
    assert reply.content == b"".join(frames)
    assert GatewayCoverage.FAILED in observe.await_args_list[2].args[0].coverage


async def test_backend_timeout_and_observer_error_are_explicit_async() -> None:
    class SlowBackend(FakeModelOnlyBackend):
        async def create_response_async(self, *, request: ModelRequest) -> bytes:
            await asyncio.sleep(0.1)
            return await super().create_response_async(request=request)

    async with _client(backend=SlowBackend(), limits=GatewayLimits(timeout_seconds=0.01)) as client:
        timeout = await client.post("/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE"}, headers=HEADERS)
    assert timeout.status_code == 504
    assert timeout.json()["error"]["code"] == "gateway_timeout"

    backend = FakeModelOnlyBackend()
    observer = AsyncMock(side_effect=RuntimeError("secret host detail"))
    async with _client(backend=backend, observer=observer) as client:
        failure = await client.post("/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE"}, headers=HEADERS)
    assert failure.status_code == 500
    assert failure.json()["error"]["code"] == "observation_failed"
    assert "secret host detail" not in failure.text
    assert backend.requests == []


async def test_stream_timeout_emits_error_and_closes_provider_async() -> None:
    class SlowStreamBackend(FakeModelOnlyBackend):
        async def stream_response_async(self, *, request: ModelRequest) -> AsyncGenerator[bytes, None]:
            self.requests.append(request)
            try:
                yield _event(name="response.created", sequence=0, response=_response(output=[], status="in_progress"))
                await asyncio.sleep(0.1)
                yield _event(name="response.completed", sequence=1, response=_response())
            finally:
                self.closed = True

    backend = SlowStreamBackend()
    observer = AsyncMock()
    async with _client(backend=backend, limits=GatewayLimits(timeout_seconds=0.01), observer=observer) as client:
        reply = await client.post(
            "/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE", "stream": True}, headers=HEADERS
        )
    assert reply.status_code == 200
    assert b"event: error\n" in reply.content
    assert b"gateway_timeout" in reply.content
    assert b"data: [DONE]" not in reply.content
    assert backend.closed
    boundary = observer.await_args_list[-1].args[0]
    assert boundary.kind == GatewayFrameKind.GATEWAY_ERROR
    assert boundary.frame in reply.content
    assert boundary.error_code == "gateway_timeout"
    assert boundary.status_code == 504
    assert GatewayCoverage.FAILED in boundary.coverage


async def test_backend_and_response_observer_exceptions_do_not_leak_host_details_async() -> None:
    class BrokenBackend(FakeModelOnlyBackend):
        async def create_response_async(self, *, request: ModelRequest) -> bytes:
            raise RuntimeError("secret provider key")

    async with _client(backend=BrokenBackend()) as client:
        failed = await client.post("/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE"}, headers=HEADERS)
    assert failed.status_code == 502
    assert failed.json()["error"]["code"] == "backend_failed"
    assert "secret provider key" not in failed.text

    observer = AsyncMock(side_effect=[None, RuntimeError("secret recorder detail")])
    async with _client(backend=FakeModelOnlyBackend(), observer=observer) as client:
        observed = await client.post("/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE"}, headers=HEADERS)
    assert observed.status_code == 500
    assert observed.json()["error"]["code"] == "observation_failed"
    assert "secret recorder detail" not in observed.text


async def test_terminal_observer_failure_does_not_mask_original_stream_error_async(
    caplog: pytest.LogCaptureFixture,
) -> None:
    created = _event(name="response.created", sequence=0, response=_response(output=[], status="in_progress"))
    backend = FakeModelOnlyBackend(streams=[[created]])
    observer = AsyncMock(side_effect=[None, None, RuntimeError("secret recorder detail")])
    async with _client(backend=backend, observer=observer) as client:
        reply = await client.post(
            "/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE", "stream": True}, headers=HEADERS
        )
    assert reply.status_code == 200
    assert reply.content.startswith(created)
    assert b"incomplete_stream" in reply.content
    assert b"observation_failed" not in reply.content
    assert b"data: [DONE]" not in reply.content
    assert "secret recorder detail" not in caplog.text
    assert "incomplete_stream" in caplog.text
    assert observer.await_count == 3
    assert observer.await_args_list[-1].args[0].kind == GatewayFrameKind.GATEWAY_ERROR
    assert observer.await_args_list[-1].args[0].error_code == "incomplete_stream"
    assert backend.closed


async def test_cancelling_guest_stream_closes_backend_without_resuming_or_refunding_async() -> None:
    class HangingBackend(FakeModelOnlyBackend):
        def __init__(self) -> None:
            super().__init__()
            self.waiting = asyncio.Event()
            self.finished = asyncio.Event()

        async def stream_response_async(self, *, request: ModelRequest) -> AsyncGenerator[bytes, None]:
            self.requests.append(request)
            try:
                yield _event(name="response.created", sequence=0, response=_response(output=[], status="in_progress"))
                self.waiting.set()
                await asyncio.Event().wait()
            finally:
                self.finished.set()

    backend = HangingBackend()
    async with _client(backend=backend, limits=GatewayLimits(max_requests=1)) as client:
        pending = asyncio.create_task(
            client.post(
                "/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE", "stream": True}, headers=HEADERS
            )
        )
        try:
            await asyncio.wait_for(backend.waiting.wait(), timeout=2)
        finally:
            pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(backend.finished.wait(), timeout=2)
        second = await client.post("/v1/responses", json={"model": ROUTE.model, "input": "OFFLINE"}, headers=HEADERS)
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "request_budget"
    assert len(backend.requests) == 1


def test_route_and_limits_reject_insecure_configuration() -> None:
    with pytest.raises(ValueError, match="guest_token"):
        GatewayRoute(run_id="run-1", model=ROUTE.model, guest_token="short")
    with pytest.raises(ValueError, match="model"):
        GatewayRoute(run_id="run-1", model="https://untrusted.invalid", guest_token=ROUTE.guest_token)
    with pytest.raises(ValueError, match="guest_token"):
        GatewayRoute(run_id="run-1", model=ROUTE.model, guest_token="é" * 32)
    with pytest.raises(ValueError, match="max_requests"):
        GatewayLimits(max_requests=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        GatewayLimits(timeout_seconds=float("inf"))
    assert ROUTE.guest_token not in repr(ROUTE)
