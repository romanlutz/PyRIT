# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# DOC501 treats the exception factory as an exception type; all failures use ModelBackendError.
# ruff: noqa: DOC501

"""Pinned, host-authenticated Responses transport; it never executes agent tools."""

import asyncio
import hmac
import json
import math
import re
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any
from urllib.parse import urlsplit

import httpx

from pyrit.prompt_target.gateway.responses_contract import (
    BackendCapabilities,
    GatewayError,
    GatewayLimits,
    GatewayRoute,
    ModelBackendError,
    ModelBackendErrorCode,
    ModelRequest,
)
from pyrit.prompt_target.gateway.responses_validation import ResponsesValidator, strict_json_loads
from pyrit.prompt_target.gateway.secret_frame_guard import CredentialEchoError, SecretFrameGuard


def _upstream_error(*, code: ModelBackendErrorCode) -> ModelBackendError:
    return ModelBackendError(code=code)


class _CredentialFrameGuard(SecretFrameGuard):
    """Parse Responses SSE frames before applying the shared credential guard."""

    def accept(self, *, frame: bytes) -> list[bytes]:
        data = self._event_data(frame=frame)
        try:
            return super().accept_payload(frame=frame, data=data)
        except (CredentialEchoError, RecursionError):
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR) from None

    def _event_data(self, *, frame: bytes) -> dict[str, Any]:
        try:
            text = frame.decode("utf-8").replace("\r\n", "\n")
            lines = text[:-2].split("\n") if text.endswith("\n\n") else []
            if len(lines) != 2 or not lines[0].startswith("event: ") or not lines[1].startswith("data: "):
                raise ValueError("Malformed SSE frame")
            data = strict_json_loads(value=lines[1][6:])
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR) from None
        if not isinstance(data, dict) or data.get("type") != lines[0][7:]:
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR)
        return data


class HttpxResponsesBackend:
    """
    Relay a model-only Responses request to a single host-owned HTTPS endpoint.

    The caller owns ``client`` and must provide a transport/client configured
    on the host. The sandbox receives neither this client nor ``auth_token``.
    Capabilities must be explicitly advertised; no provider support is guessed.
    Only raw, bounded model response bytes leave this class. Tool calls and
    tool results are never interpreted or executed here.
    """

    _SSE_BOUNDARY = re.compile(rb"\r?\n\r?\n")
    _MAX_EMPTY_CHUNKS = 16
    _YIELD_EVERY_FRAMES = 128

    def __init__(
        self,
        *,
        route: GatewayRoute,
        endpoint: str,
        auth_token: str,
        client: httpx.AsyncClient,
        limits: GatewayLimits,
        capabilities: BackendCapabilities,
        timeout_seconds: float | None = None,
    ) -> None:
        """
        Bind one run to a fixed upstream endpoint and host-only credential.

        Args:
            route (GatewayRoute): Run and model alias already bound by the gateway.
            endpoint (str): Full HTTPS Responses URL; never taken from the guest.
            auth_token (str): Host-owned bearer token, distinct from the guest token.
            client (httpx.AsyncClient): Host-owned injected client; caller manages its lifetime.
            limits (GatewayLimits): The same byte/time ceilings used by the gateway.
            capabilities (BackendCapabilities): Features known to work on this provider.
            timeout_seconds (float | None): Optional upstream deadline within the gateway deadline.

        Raises:
            ValueError: If routing, endpoint, authentication, or deadline is unsafe.
        """
        self._endpoint = self._validate_endpoint(endpoint=endpoint)
        if (
            not isinstance(auth_token, str)
            or not 16 <= len(auth_token) <= 4096
            or not auth_token.isascii()
            or any(not 33 <= ord(character) <= 126 for character in auth_token)
            or hmac.compare_digest(auth_token, route.guest_token)
        ):
            raise ValueError("Host model auth must be a distinct, nonempty ASCII bearer token")
        if not isinstance(client, httpx.AsyncClient):
            raise ValueError("An injected host-owned httpx.AsyncClient is required")
        if not isinstance(capabilities, BackendCapabilities) or any(
            type(getattr(capabilities, name)) is not bool
            for name in ("streaming", "function_tools", "custom_tools", "reasoning")
        ):
            raise ValueError("Explicit boolean model backend capabilities are required")
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= limits.timeout_seconds
        ):
            raise ValueError("Upstream timeout must be positive and no longer than the gateway deadline")
        self.capabilities = capabilities
        self._route = route
        self._auth_token = auth_token
        self._client = client
        self._limits = limits
        self._timeout_seconds = timeout_seconds if timeout_seconds is not None else limits.timeout_seconds
        self._validator = ResponsesValidator(route=route, limits=limits, capabilities=capabilities)

    async def create_response_async(self, *, request: ModelRequest) -> bytes:
        """
        Return bounded original JSON bytes without copying host auth to the guest.

        Args:
            request (ModelRequest): Gateway-validated text/tool Responses request.

        Returns:
            bytes: Exact upstream Responses JSON body.

        Raises:
            ModelBackendError: If HTTP, transport, deadline, size, or content checks fail.
        """
        body = self._request_bytes(request=request, streaming=False)
        deadline = asyncio.get_running_loop().time() + self._timeout_seconds
        response = await self._open_async(body=body, streaming=False, deadline=deadline)
        try:
            result = bytearray()
            iterator = self._response_chunks_async(response=response)
            empty_chunks = 0
            while True:
                try:
                    chunk = await self._next_chunk_async(iterator=iterator, deadline=deadline)
                except StopAsyncIteration:
                    break
                if not chunk:
                    empty_chunks += 1
                    if empty_chunks > self._MAX_EMPTY_CHUNKS:
                        raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR)
                    continue
                empty_chunks = 0
                if len(result) + len(chunk) > self._limits.max_response_bytes:
                    raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR)
                result.extend(chunk)
            raw = bytes(result)
            self._validate_body(raw=raw)
            if asyncio.get_running_loop().time() >= deadline:
                raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_TIMEOUT)
            return raw
        finally:
            await response.aclose()

    async def stream_response_async(self, *, request: ModelRequest) -> AsyncGenerator[bytes, None]:
        """
        Yield complete, unchanged upstream SSE frames until a real [DONE].

        Args:
            request (ModelRequest): Gateway-validated streaming Responses request.

        Yields:
            bytes: Exact upstream SSE frames in their original order.

        Raises:
            ModelBackendError: If upstream streaming, auth protection, or limits fail.
        """
        if not self.capabilities.streaming:
            raise NotImplementedError("Model backend does not advertise Responses SSE streaming")
        body = self._request_bytes(request=request, streaming=True)
        deadline = asyncio.get_running_loop().time() + self._timeout_seconds
        response = await self._open_async(body=body, streaming=True, deadline=deadline)
        guard = _CredentialFrameGuard(token=self._auth_token)
        buffer = bytearray()
        total_bytes = 0
        empty_chunks = 0
        frames_seen = 0
        try:
            iterator = self._response_chunks_async(response=response)
            while True:
                try:
                    chunk = await self._next_chunk_async(iterator=iterator, deadline=deadline)
                except StopAsyncIteration:
                    raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR) from None
                if not chunk:
                    empty_chunks += 1
                    if empty_chunks > self._MAX_EMPTY_CHUNKS:
                        raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR)
                    continue
                empty_chunks = 0
                total_bytes += len(chunk)
                if total_bytes > self._limits.max_response_bytes:
                    raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR)
                buffer.extend(chunk)
                while match := self._SSE_BOUNDARY.search(buffer):
                    frames_seen += 1
                    if frames_seen % self._YIELD_EVERY_FRAMES == 0:
                        await asyncio.sleep(0)
                    if asyncio.get_running_loop().time() >= deadline:
                        raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_TIMEOUT)
                    frame = bytes(buffer[: match.end()])
                    del buffer[: match.end()]
                    if frame.replace(b"\r\n", b"\n") == b"data: [DONE]\n\n":
                        if buffer:
                            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR)
                        for pending in guard.finish():
                            yield pending
                        yield frame
                        return
                    for safe_frame in guard.accept(frame=frame):
                        yield safe_frame
        finally:
            await response.aclose()

    def _request_bytes(self, *, request: ModelRequest, streaming: bool) -> bytes:
        if request.run_id != self._route.run_id or request.body.get("model") != self._route.model:
            raise ValueError("Model request does not match this host-owned run route")
        try:
            validated = self._validator.validate_request(body=request.body)
        except GatewayError:
            raise ValueError("Model request violates the gateway's supported Responses subset") from None
        if validated.streaming != streaming or validated.output_token_limit != request.output_token_limit:
            raise ValueError("Model request stream or token budget does not match the gateway route")
        raw = json.dumps(validated.body, allow_nan=False, separators=(",", ":")).encode()
        if len(raw) > self._limits.max_request_bytes:
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR)
        return raw

    async def _open_async(self, *, body: bytes, streaming: bool, deadline: float) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self._auth_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if streaming else "application/json",
            "Accept-Encoding": "identity",
        }
        http_request = httpx.Request("POST", self._endpoint, headers=headers, content=body)
        try:
            async with asyncio.timeout_at(deadline):
                response = await self._client.send(http_request, stream=True, auth=None, follow_redirects=False)
        except (TimeoutError, httpx.TimeoutException):
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_TIMEOUT) from None
        except httpx.DecodingError:
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR) from None
        except httpx.RequestError:
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_NETWORK_ERROR) from None
        try:
            self._validate_headers(response=response, streaming=streaming)
        except ModelBackendError:
            await response.aclose()
            raise
        return response

    async def _response_chunks_async(self, *, response: httpx.Response) -> AsyncGenerator[bytes, None]:
        if response.is_stream_consumed:
            yield response.content
            return
        async for chunk in response.aiter_raw():
            yield chunk

    def _validate_headers(self, *, response: httpx.Response, streaming: bool) -> None:
        if response.status_code != 200:
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_HTTP_ERROR)
        expected = "text/event-stream" if streaming else "application/json"
        if response.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower() != expected:
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR)
        if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR)
        lengths = response.headers.get_list("content-length")
        if len(lengths) > 1 or (
            lengths and (len(lengths[0]) > 20 or not lengths[0].isascii() or not lengths[0].isdigit())
        ):
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR)
        if lengths and int(lengths[0]) > self._limits.max_response_bytes:
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR)

    async def _next_chunk_async(self, *, iterator: AsyncIterator[bytes], deadline: float) -> bytes:
        try:
            async with asyncio.timeout_at(deadline):
                chunk = await anext(iterator)
                await asyncio.sleep(0)
                return chunk
        except StopAsyncIteration:
            raise
        except (TimeoutError, httpx.TimeoutException):
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_TIMEOUT) from None
        except httpx.DecodingError:
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR) from None
        except httpx.RequestError:
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_NETWORK_ERROR) from None
        except httpx.StreamError:
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR) from None

    def _validate_body(self, *, raw: bytes) -> None:
        if not raw or self._auth_token.encode() in raw:
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR)
        try:
            parsed = strict_json_loads(value=raw)
            if not isinstance(parsed, dict):
                raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR)
            SecretFrameGuard(token=self._auth_token).check_body(raw=raw, parsed=parsed)
        except CredentialEchoError:
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR) from None
        except (ValueError, RecursionError):
            raise _upstream_error(code=ModelBackendErrorCode.UPSTREAM_STREAM_ERROR) from None

    @staticmethod
    def _validate_endpoint(*, endpoint: str) -> httpx.URL:
        if not isinstance(endpoint, str) or not endpoint.isascii() or any(ord(char) <= 32 for char in endpoint):
            raise ValueError("A pinned HTTPS Responses endpoint is required")
        try:
            parsed = urlsplit(endpoint)
            path = parsed.path
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or parsed.port == 0
                or not path.endswith("/responses")
                or any(segment in {"", ".", ".."} for segment in path[1:].split("/"))
            ):
                raise ValueError("Invalid Responses endpoint")
            return httpx.URL(endpoint)
        except ValueError:
            raise ValueError(
                "A pinned HTTPS Responses endpoint without query, fragment, or userinfo is required"
            ) from None
