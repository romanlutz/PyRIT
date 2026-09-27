# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# DOC501 mistakes the fixed-error factory for a distinct exception type.
# ruff: noqa: DOC501

"""Host-pinned Anthropic Messages HTTP transport with no tool execution."""

import asyncio
import hmac
import math
import re
from collections.abc import AsyncGenerator, AsyncIterator
from urllib.parse import urlsplit

import httpx

from pyrit.prompt_target.gateway.json_utility import strict_json_loads
from pyrit.prompt_target.gateway.messages_contract import (
    MessagesBackendError,
    MessagesBackendErrorCode,
    MessagesCapabilities,
    MessagesProviderError,
    MessagesRequest,
    MessagesResponse,
    MessagesStream,
)
from pyrit.prompt_target.gateway.messages_stream_validation import parse_messages_sse_frame
from pyrit.prompt_target.gateway.messages_validation import MessagesValidator
from pyrit.prompt_target.gateway.responses_contract import GatewayError, GatewayLimits, GatewayRoute
from pyrit.prompt_target.gateway.secret_frame_guard import CredentialEchoError, SecretFrameGuard


def _backend_error(*, code: MessagesBackendErrorCode) -> MessagesBackendError:
    return MessagesBackendError(code=code)


class HttpxMessagesBackend:
    """Relay original Anthropic Messages bytes to one host-configured upstream."""

    _SSE_BOUNDARY = re.compile(rb"\r?\n\r?\n")
    _MAX_EMPTY_CHUNKS = 16
    _YIELD_EVERY_FRAMES = 128

    def __init__(
        self,
        *,
        route: GatewayRoute,
        endpoint: str,
        host_api_key: str,
        client: httpx.AsyncClient,
        limits: GatewayLimits,
        capabilities: MessagesCapabilities,
        timeout_seconds: float | None = None,
    ) -> None:
        """
        Bind the run to a pinned HTTPS `/v1/messages` endpoint and host-only key.

        Args:
            route (GatewayRoute): One run and the exact upstream Claude model ID.
            endpoint (str): Trusted, pinned Anthropic-format HTTPS Messages URL.
            host_api_key (str): Host-held upstream key, never passed to the sandbox.
            client (httpx.AsyncClient): Injected host client; the caller owns its lifetime.
            limits (GatewayLimits): The same byte/time/token ceilings as the ASGI gateway.
            capabilities (MessagesCapabilities): Explicitly verified upstream features/betas.
            timeout_seconds (float | None): Optional upstream deadline within the gateway deadline.

        Raises:
            ValueError: If the upstream route, key, client, or limits are not safe.
        """
        self._endpoint = self._validate_endpoint(endpoint=endpoint)
        if (
            not isinstance(host_api_key, str)
            or not 16 <= len(host_api_key) <= 4096
            or not host_api_key.isascii()
            or any(not 33 <= ord(character) <= 126 for character in host_api_key)
            or hmac.compare_digest(host_api_key, route.guest_token)
        ):
            raise ValueError("Host API key must be distinct, nonempty, and ASCII-only")
        if not isinstance(client, httpx.AsyncClient):
            raise ValueError("An injected host-owned httpx.AsyncClient is required")
        if not isinstance(capabilities, MessagesCapabilities):
            raise ValueError("Explicit Messages capabilities are required")
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= limits.timeout_seconds
        ):
            raise ValueError("Upstream timeout must be positive and no longer than the gateway deadline")
        self.capabilities = capabilities
        self._route = route
        self._host_api_key = host_api_key
        self._client = client
        self._limits = limits
        self._timeout_seconds = timeout_seconds if timeout_seconds is not None else limits.timeout_seconds
        self._validator = MessagesValidator(route=route, limits=limits, capabilities=capabilities)

    async def create_message_async(self, *, request: MessagesRequest) -> MessagesResponse:
        """
        Return exact model JSON and selected safe headers, never executing tools.

        Args:
            request (MessagesRequest): Validated original Anthropic body and approved headers.

        Returns:
            MessagesResponse: Original provider JSON body, status, and safe headers.

        Raises:
            MessagesBackendError: If HTTP transport, deadline, or provider data is unsafe.
            MessagesProviderError: If the upstream returned a genuine safe Anthropic error.
        """
        self._validate_request(request=request, streaming=False)
        deadline = asyncio.get_running_loop().time() + self._timeout_seconds
        response = await self._open_async(request=request, deadline=deadline, streaming=False)
        try:
            body = await self._read_body_async(response=response, deadline=deadline)
            self._guard_body(body=body)
            return MessagesResponse(status_code=200, body=body, headers=self._selected_headers(response=response))
        finally:
            await response.aclose()

    async def open_stream_async(self, *, request: MessagesRequest) -> MessagesStream:
        """
        Open a real SSE response and return its closeable, unbuffered frame iterator.

        Args:
            request (MessagesRequest): Validated streaming Anthropic body and approved headers.

        Returns:
            MessagesStream: Original provider frames and close operation.

        Raises:
            MessagesBackendError: If the upstream cannot stream these Messages.
            MessagesProviderError: If the provider returned a genuine safe HTTP error.
        """
        if not self.capabilities.streaming:
            raise NotImplementedError("Model backend does not advertise Anthropic streaming")
        self._validate_request(request=request, streaming=True)
        deadline = asyncio.get_running_loop().time() + self._timeout_seconds
        response = await self._open_async(request=request, deadline=deadline, streaming=True)
        try:
            return MessagesStream(
                frames=self._iter_frames_async(response=response, deadline=deadline),
                headers=self._selected_headers(response=response),
                _close=response.aclose,
            )
        except MessagesBackendError:
            await response.aclose()
            raise

    def _validate_request(self, *, request: MessagesRequest, streaming: bool) -> None:
        if not isinstance(request, MessagesRequest) or request.run_id != self._route.run_id:
            raise ValueError("Messages request is not bound to this host-owned run")
        if request.query_string not in (b"", b"beta=true"):
            raise ValueError("Messages query must be the documented beta=true flag")
        if not isinstance(request.body_bytes, bytes) or len(request.body_bytes) > self._limits.max_request_bytes:
            raise ValueError("Messages request body exceeds the host byte limit")
        try:
            body = strict_json_loads(value=request.body_bytes)
        except (ValueError, RecursionError):
            raise ValueError("Messages body is not strict JSON") from None
        if not isinstance(body, dict) or body != request.body:
            raise ValueError("Messages body differs from its original wire bytes")
        try:
            self._validator.validate_headers(version=request.anthropic_version, beta=request.anthropic_beta)
            checked = self._validator.validate_request(body=body)
        except GatewayError:
            raise ValueError("Messages request violates the backend's verified capabilities") from None
        if (
            type(request.max_tokens) is not int
            or type(request.streaming) is not bool
            or request.max_tokens != checked.max_tokens
            or request.streaming != streaming
            or checked.streaming != streaming
            or type(request.advertised_tools) is not frozenset
            or request.advertised_tools != checked.advertised_tools
        ):
            raise ValueError("Messages output-token budget or stream mode differs from the gateway")

    async def _open_async(self, *, request: MessagesRequest, deadline: float, streaming: bool) -> httpx.Response:
        headers = {
            "x-api-key": self._host_api_key,
            "anthropic-version": request.anthropic_version,
            "content-type": "application/json",
            "accept": "text/event-stream" if streaming else "application/json",
            "accept-encoding": "identity",
        }
        if request.anthropic_beta is not None:
            headers["anthropic-beta"] = request.anthropic_beta
        url = self._endpoint.copy_with(query=request.query_string) if request.query_string else self._endpoint
        http_request = httpx.Request(
            "POST",
            url,
            headers=headers,
            content=request.body_bytes,
        )
        try:
            async with asyncio.timeout_at(deadline):
                response = await self._client.send(http_request, stream=True, auth=None, follow_redirects=False)
        except (TimeoutError, httpx.TimeoutException):
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_TIMEOUT) from None
        except httpx.DecodingError:
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR) from None
        except httpx.RequestError:
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_NETWORK_ERROR) from None
        try:
            self._validate_response_headers(response=response, streaming=streaming)
            if response.status_code != 200:
                error_body = await self._read_body_async(response=response, deadline=deadline)
                self._guard_body(body=error_body)
                self._validator.validate_provider_error(value=strict_json_loads(value=error_body))
                raise MessagesProviderError(
                    response=MessagesResponse(
                        status_code=response.status_code,
                        body=error_body,
                        headers=self._selected_headers(response=response),
                    )
                )
        except MessagesProviderError:
            await response.aclose()
            raise
        except (GatewayError, ValueError, RecursionError):
            await response.aclose()
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR) from None
        except MessagesBackendError:
            await response.aclose()
            raise
        return response

    def _validate_response_headers(self, *, response: httpx.Response, streaming: bool) -> None:
        if response.status_code != 200 and not 400 <= response.status_code <= 599:
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
        content_types = response.headers.get_list("content-type")
        expected = "text/event-stream" if streaming and response.status_code == 200 else "application/json"
        if len(content_types) != 1 or content_types[0].split(";", maxsplit=1)[0].strip().lower() != expected:
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
        if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
        lengths = response.headers.get_list("content-length")
        if len(lengths) > 1 or (
            lengths and (len(lengths[0]) > 20 or not lengths[0].isascii() or not lengths[0].isdigit())
        ):
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
        if lengths and int(lengths[0]) > self._limits.max_response_bytes:
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)

    def _selected_headers(self, *, response: httpx.Response) -> tuple[tuple[str, str], ...]:
        selected: list[tuple[str, str]] = []
        for name, value in response.headers.items():
            normalized = name.lower()
            if normalized not in {"content-type", "retry-after", "x-should-retry"} and not normalized.startswith(
                "anthropic-ratelimit-unified-"
            ):
                continue
            if (
                len(value) > 512
                or not value.isascii()
                or any(ord(character) < 32 or ord(character) > 126 for character in value)
                or self._host_api_key in value
            ):
                raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
            if normalized == "retry-after" and (not value.isdecimal() or len(value) > 5):
                raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
            if normalized == "x-should-retry" and value not in ("true", "false"):
                raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
            selected.append((name, value))
            if len(selected) > 64:
                raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
        if self._host_api_key in "".join(value for _, value in selected):
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
        return tuple(selected)

    async def _read_body_async(self, *, response: httpx.Response, deadline: float) -> bytes:
        body = bytearray()
        empty_chunks = 0
        iterator = self._response_chunks_async(response=response)
        while True:
            try:
                chunk = await self._next_chunk_async(iterator=iterator, deadline=deadline)
            except StopAsyncIteration:
                break
            if not chunk:
                empty_chunks += 1
                if empty_chunks > self._MAX_EMPTY_CHUNKS:
                    raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
                continue
            empty_chunks = 0
            if len(body) + len(chunk) > self._limits.max_response_bytes:
                raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
            body.extend(chunk)
        if asyncio.get_running_loop().time() >= deadline:
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_TIMEOUT)
        return bytes(body)

    async def _iter_frames_async(self, *, response: httpx.Response, deadline: float) -> AsyncGenerator[bytes, None]:
        buffer = bytearray()
        total_bytes = 0
        empty_chunks = 0
        frames_seen = 0
        terminal = False
        guard = SecretFrameGuard(token=self._host_api_key)
        iterator = self._response_chunks_async(response=response)
        try:
            while not terminal:
                try:
                    chunk = await self._next_chunk_async(iterator=iterator, deadline=deadline)
                except StopAsyncIteration:
                    raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR) from None
                if not chunk:
                    empty_chunks += 1
                    if empty_chunks > self._MAX_EMPTY_CHUNKS:
                        raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
                    continue
                empty_chunks = 0
                total_bytes += len(chunk)
                if total_bytes > self._limits.max_response_bytes:
                    raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
                buffer.extend(chunk)
                while match := self._SSE_BOUNDARY.search(buffer):
                    frames_seen += 1
                    if frames_seen % self._YIELD_EVERY_FRAMES == 0:
                        await asyncio.sleep(0)
                    if asyncio.get_running_loop().time() >= deadline:
                        raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_TIMEOUT)
                    frame = bytes(buffer[: match.end()])
                    del buffer[: match.end()]
                    try:
                        parsed = parse_messages_sse_frame(frame=frame)
                        ready = guard.accept_payload(frame=frame, data=parsed.credential_scan_data)
                    except (GatewayError, CredentialEchoError, RecursionError):
                        raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR) from None
                    terminal = parsed.name in ("message_stop", "error")
                    if terminal and buffer:
                        raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR)
                    for original in ready:
                        yield original
            for original in guard.finish():
                yield original
        finally:
            await response.aclose()

    async def _response_chunks_async(self, *, response: httpx.Response) -> AsyncGenerator[bytes, None]:
        if response.is_stream_consumed:
            yield response.content
            return
        async for chunk in response.aiter_raw():
            yield chunk

    async def _next_chunk_async(self, *, iterator: AsyncIterator[bytes], deadline: float) -> bytes:
        try:
            async with asyncio.timeout_at(deadline):
                chunk = await anext(iterator)
                await asyncio.sleep(0)
                return chunk
        except StopAsyncIteration:
            raise
        except (TimeoutError, httpx.TimeoutException):
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_TIMEOUT) from None
        except httpx.DecodingError:
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR) from None
        except httpx.RequestError:
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_NETWORK_ERROR) from None
        except httpx.StreamError:
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR) from None

    def _guard_body(self, *, body: bytes) -> None:
        try:
            parsed = strict_json_loads(value=body)
            SecretFrameGuard(token=self._host_api_key).check_body(raw=body, parsed=parsed)
        except (ValueError, RecursionError, CredentialEchoError):
            raise _backend_error(code=MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR) from None

    @staticmethod
    def _validate_endpoint(*, endpoint: str) -> httpx.URL:
        if not isinstance(endpoint, str) or not endpoint.isascii() or any(ord(char) <= 32 for char in endpoint):
            raise ValueError("A pinned HTTPS Anthropic Messages endpoint is required")
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
                or not path.endswith("/v1/messages")
                or "%" in path
                or any(segment in {"", ".", ".."} for segment in path[1:].split("/"))
            ):
                raise ValueError("Invalid Messages endpoint")
            return httpx.URL(endpoint)
        except ValueError:
            raise ValueError("A pinned HTTPS Messages endpoint without query or userinfo is required") from None
