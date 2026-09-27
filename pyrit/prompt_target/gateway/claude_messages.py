# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One-run Anthropic Messages ASGI gateway for a sandboxed Claude Code CLI."""

import asyncio
import hmac
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from pyrit.prompt_target.gateway.json_utility import strict_json_loads
from pyrit.prompt_target.gateway.messages_contract import (
    MessagesBackendError,
    MessagesCapabilities,
    MessagesCoverage,
    MessagesObservation,
    MessagesObservationCallback,
    MessagesProviderError,
    MessagesRequest,
    MessagesResponse,
    MessagesStream,
    ModelOnlyMessagesBackend,
)
from pyrit.prompt_target.gateway.messages_stream_validation import MessagesStreamValidator
from pyrit.prompt_target.gateway.messages_validation import MessagesValidator
from pyrit.prompt_target.gateway.responses_contract import GatewayError, GatewayFrameKind, GatewayLimits, GatewayRoute

logger = logging.getLogger(__name__)


def _error_body(*, error: GatewayError) -> dict[str, Any]:
    if error.status_code == 401:
        kind = "authentication_error"
    elif error.status_code == 403:
        kind = "permission_error"
    elif error.status_code == 429:
        kind = "rate_limit_error"
    elif error.status_code < 500:
        kind = "invalid_request_error"
    else:
        kind = "api_error"
    return {"type": "error", "error": {"type": kind, "message": f"{error.code}: {error}"}}


def _gateway_json_error(*, error: GatewayError) -> JSONResponse:
    headers = {"Cache-Control": "no-store", "X-Should-Retry": "false"}
    if error.status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(_error_body(error=error), status_code=error.status_code, headers=headers)


def _gateway_sse_error(*, error: GatewayError) -> bytes:
    data = json.dumps(_error_body(error=error), separators=(",", ":"))
    return f"event: error\ndata: {data}\n\n".encode()


def _safe_backend_error(*, error: MessagesBackendError) -> GatewayError:
    return GatewayError(status_code=error.status_code, code=error.code.value, message=str(error))


class _ClaudeMessagesGateway:
    """Authenticate one run, route one model, and never execute a client tool."""

    _IGNORED_HEADERS = frozenset(
        {
            "host",
            "accept",
            "accept-encoding",
            "user-agent",
            "connection",
            "content-length",
            "content-type",
            "authorization",
            "anthropic-version",
            "anthropic-beta",
            "x-pyrit-run-id",
        }
    )

    def __init__(
        self,
        *,
        route: GatewayRoute,
        limits: GatewayLimits,
        backend: ModelOnlyMessagesBackend | None,
        observation_callback: MessagesObservationCallback | None,
    ) -> None:
        self._route = route
        self._limits = limits
        self._backend = backend
        self._observation_callback = observation_callback
        self._validator = MessagesValidator(
            route=route, limits=limits, capabilities=backend.capabilities if backend else MessagesCapabilities()
        )
        self._lock = asyncio.Lock()
        self._requests = 0
        self._reserved_request_bytes = 0
        self._reserved_output_tokens = 0

    async def _handle_async(self, request: Request) -> Response:
        deadline = asyncio.get_running_loop().time() + self._limits.timeout_seconds
        model_request: MessagesRequest | None = None
        dispatched = False
        try:
            self._authenticate(request=request)
            version, beta = self._headers(request=request)
            query = request.scope.get("query_string", b"")
            if query not in (b"", b"beta=true"):
                raise GatewayError(status_code=501, code="unsupported_feature", message="Unsupported Messages query")
            raw = await self._read_body_async(request=request, deadline=deadline)
            body = self._decode_body(raw=raw)
            self._validator.validate_headers(version=version, beta=beta)
            validated = self._validator.validate_request(body=body)
            if self._backend is None:
                raise GatewayError(
                    status_code=501, code="model_backend_required", message="A model-only Messages backend is required"
                )
            await self._reserve_async(tokens=validated.max_tokens, request_bytes=len(raw))
            model_request = MessagesRequest(
                run_id=self._route.run_id,
                request_id=uuid4().hex,
                body_bytes=raw,
                body=body,
                anthropic_version=version,
                anthropic_beta=beta,
                query_string=query,
                max_tokens=validated.max_tokens,
                streaming=validated.streaming,
                advertised_tools=validated.advertised_tools,
            )
            await self._observe_async(
                request=model_request,
                kind=GatewayFrameKind.REQUEST,
                frame=raw,
                coverage=validated.coverage,
                headers=self._forwarded_headers(request=model_request),
                deadline=deadline,
            )
            dispatched = True
            if validated.streaming:
                return await self._start_stream_async(request=model_request, deadline=deadline)
            return await self._complete_async(request=model_request, deadline=deadline)
        except MessagesProviderError as error:
            if model_request is None:
                raise
            return await self._provider_error_async(request=model_request, response=error.response, deadline=deadline)
        except GatewayError as error:
            response = _gateway_json_error(error=error)
            if model_request is not None and dispatched:
                await self._record_gateway_error_async(
                    request=model_request, error=error, frame=bytes(response.body), streaming=model_request.streaming
                )
            return response

    def _authenticate(self, *, request: Request) -> None:
        credentials = request.headers.getlist("authorization")
        if len(credentials) != 1 or not credentials[0].startswith("Bearer "):
            raise GatewayError(status_code=401, code="invalid_token", message="A run-scoped bearer token is required")
        token = credentials[0][7:]
        if (
            len(token) != len(self._route.guest_token)
            or not token.isascii()
            or not hmac.compare_digest(token, self._route.guest_token)
        ):
            raise GatewayError(status_code=401, code="invalid_token", message="Invalid run-scoped bearer token")
        runs = request.headers.getlist("x-pyrit-run-id")
        if len(runs) != 1 or runs[0] != self._route.run_id:
            raise GatewayError(status_code=403, code="invalid_run", message="Request is not routed to this run")

    def _headers(self, *, request: Request) -> tuple[str, str | None]:
        raw_headers = request.scope.get("headers", [])
        if len(raw_headers) > 64 or any(len(name) > 128 or len(value) > 4096 for name, value in raw_headers):
            raise GatewayError(status_code=400, code="invalid_headers", message="Messages headers exceeded limits")
        for name, _ in raw_headers:
            normalized = name.decode("latin-1").lower()
            if normalized not in self._IGNORED_HEADERS and not normalized.startswith(
                ("x-stainless-", "x-claude-code-")
            ):
                raise GatewayError(status_code=501, code="unsupported_feature", message="Unsupported Messages header")
        versions = request.headers.getlist("anthropic-version")
        betas = request.headers.getlist("anthropic-beta")
        if len(versions) != 1 or len(betas) > 1:
            raise GatewayError(
                status_code=400, code="invalid_headers", message="Messages version or beta header is ambiguous"
            )
        if len(request.headers.getlist("content-type")) != 1:
            raise GatewayError(status_code=415, code="invalid_content_type", message="Content-Type is required")
        if request.headers["content-type"].split(";", maxsplit=1)[0].strip().lower() != "application/json":
            raise GatewayError(
                status_code=415, code="invalid_content_type", message="Content-Type must be application/json"
            )
        return versions[0], betas[0] if betas else None

    async def _read_body_async(self, *, request: Request, deadline: float) -> bytes:
        lengths = request.headers.getlist("content-length")
        if len(lengths) > 1 or (
            lengths and (len(lengths[0]) > 20 or not lengths[0].isascii() or not lengths[0].isdigit())
        ):
            raise GatewayError(status_code=400, code="invalid_content_length", message="Invalid Content-Length")
        if lengths and int(lengths[0]) > self._limits.max_request_bytes:
            raise GatewayError(
                status_code=413, code="request_too_large", message="Messages request exceeded byte limit"
            )
        collected = bytearray()
        try:
            async with asyncio.timeout_at(deadline):
                async for chunk in request.stream():
                    if len(collected) + len(chunk) > self._limits.max_request_bytes:
                        raise GatewayError(
                            status_code=413, code="request_too_large", message="Messages request exceeded byte limit"
                        )
                    collected.extend(chunk)
        except TimeoutError as exc:
            raise GatewayError(status_code=504, code="gateway_timeout", message="Messages request timed out") from exc
        return bytes(collected)

    def _decode_body(self, *, raw: bytes) -> dict[str, Any]:
        try:
            body = strict_json_loads(value=raw)
        except (ValueError, RecursionError) as exc:
            raise GatewayError(
                status_code=400, code="invalid_json", message="Messages body must be valid JSON"
            ) from exc
        if not isinstance(body, dict):
            raise GatewayError(status_code=400, code="invalid_json", message="Messages body must be a JSON object")
        return body

    async def _reserve_async(self, *, tokens: int, request_bytes: int) -> None:
        async with self._lock:
            if self._requests >= self._limits.max_requests:
                raise GatewayError(status_code=429, code="request_budget", message="Run request budget exhausted")
            if self._reserved_request_bytes + request_bytes > self._limits.max_total_request_bytes:
                raise GatewayError(status_code=429, code="input_byte_budget", message="Run input-byte budget exhausted")
            if self._reserved_output_tokens + tokens > self._limits.max_total_output_tokens:
                raise GatewayError(status_code=429, code="token_budget", message="Run output-token budget exhausted")
            self._requests += 1
            self._reserved_request_bytes += request_bytes
            self._reserved_output_tokens += tokens

    def _forwarded_headers(self, *, request: MessagesRequest) -> tuple[tuple[str, str], ...]:
        version = (("anthropic-version", request.anthropic_version),)
        if request.anthropic_beta is None:
            return version
        return (*version, ("anthropic-beta", request.anthropic_beta))

    async def _observe_async(
        self,
        *,
        request: MessagesRequest,
        kind: GatewayFrameKind,
        frame: bytes,
        coverage: frozenset[MessagesCoverage],
        deadline: float,
        headers: tuple[tuple[str, str], ...] = (),
        error: GatewayError | None = None,
        status_code: int | None = None,
    ) -> None:
        if self._observation_callback is None:
            return
        observation = MessagesObservation(
            run_id=request.run_id,
            request_id=request.request_id,
            kind=kind,
            frame=frame,
            coverage=coverage,
            headers=headers,
            query_string=request.query_string.decode(),
            error_code=error.code if error is not None else None,
            status_code=status_code if status_code is not None else (error.status_code if error else None),
        )
        try:
            async with asyncio.timeout_at(deadline):
                await self._observation_callback(observation)
        except TimeoutError as exc:
            raise GatewayError(
                status_code=504, code="gateway_timeout", message="Messages observation timed out"
            ) from exc
        except Exception as exc:
            logger.error("Messages observation failed (%s)", type(exc).__name__)
            raise GatewayError(
                status_code=500, code="observation_failed", message="Host observation callback failed"
            ) from exc

    async def _record_gateway_error_async(
        self, *, request: MessagesRequest, error: GatewayError, frame: bytes, streaming: bool
    ) -> None:
        coverage = {MessagesCoverage.FAILED}
        if streaming:
            coverage.add(MessagesCoverage.STREAMING)
        try:
            await self._observe_async(
                request=request,
                kind=GatewayFrameKind.GATEWAY_ERROR,
                frame=frame,
                coverage=frozenset(coverage),
                error=error,
                deadline=asyncio.get_running_loop().time() + min(1.0, self._limits.timeout_seconds),
            )
        except GatewayError as observation_error:
            logger.error("Could not record Messages failure %s (%s)", error.code, observation_error.code)

    async def _provider_error_async(
        self, *, request: MessagesRequest, response: MessagesResponse, deadline: float
    ) -> Response:
        try:
            if (
                not 400 <= response.status_code <= 599
                or not isinstance(response.body, bytes)
                or len(response.body) > self._limits.max_response_bytes
            ):
                raise GatewayError(
                    status_code=502, code="invalid_backend_response", message="Provider error body is invalid"
                )
            payload = strict_json_loads(value=response.body)
            self._validator.validate_provider_headers(headers=response.headers, streaming=False)
            self._validator.validate_provider_error(value=payload)
        except (ValueError, RecursionError, GatewayError):
            error = GatewayError(
                status_code=502, code="invalid_backend_response", message="Provider error body is invalid"
            )
            rendered = _gateway_json_error(error=error)
            await self._record_gateway_error_async(
                request=request, error=error, frame=bytes(rendered.body), streaming=request.streaming
            )
            return rendered
        try:
            await self._observe_async(
                request=request,
                kind=GatewayFrameKind.RESPONSE,
                frame=response.body,
                coverage=frozenset({MessagesCoverage.FAILED}),
                status_code=response.status_code,
                headers=response.headers,
                deadline=deadline,
            )
        except GatewayError as observation_error:
            logger.error("Could not record provider Messages error (%s)", observation_error.code)
        return Response(content=response.body, status_code=response.status_code, headers=dict(response.headers))

    async def _complete_async(self, *, request: MessagesRequest, deadline: float) -> Response:
        if self._backend is None:
            raise GatewayError(status_code=501, code="model_backend_required", message="No model backend is configured")
        try:
            async with asyncio.timeout_at(deadline):
                reply = await self._backend.create_message_async(request=request)
        except TimeoutError as exc:
            raise GatewayError(status_code=504, code="gateway_timeout", message="Model response timed out") from exc
        except MessagesBackendError as exc:
            raise _safe_backend_error(error=exc) from exc
        except MessagesProviderError:
            raise
        except Exception as exc:
            logger.error("Host-only Messages backend failed (%s)", type(exc).__name__)
            raise GatewayError(
                status_code=502, code="backend_failed", message="Host-only Messages backend failed"
            ) from exc
        if reply.status_code != 200:
            return await self._provider_error_async(request=request, response=reply, deadline=deadline)
        if not isinstance(reply.body, bytes) or len(reply.body) > self._limits.max_response_bytes:
            raise GatewayError(status_code=502, code="response_too_large", message="Model response exceeded byte limit")
        self._validator.validate_provider_headers(headers=reply.headers, streaming=False)
        try:
            body = strict_json_loads(value=reply.body)
        except (ValueError, RecursionError) as exc:
            raise GatewayError(
                status_code=502, code="invalid_backend_response", message="Model returned invalid JSON"
            ) from exc
        coverage = self._validator.validate_provider_message(
            value=body,
            max_tokens=request.max_tokens,
            advertised_tools=request.advertised_tools,
        )
        await self._observe_async(
            request=request,
            kind=GatewayFrameKind.RESPONSE,
            frame=reply.body,
            coverage=coverage,
            headers=reply.headers,
            status_code=200,
            deadline=deadline,
        )
        return Response(content=reply.body, status_code=200, headers=dict(reply.headers))

    async def _start_stream_async(self, *, request: MessagesRequest, deadline: float) -> Response:
        if self._backend is None:
            raise GatewayError(status_code=501, code="model_backend_required", message="No model backend is configured")
        try:
            async with asyncio.timeout_at(deadline):
                stream = await self._backend.open_stream_async(request=request)
        except TimeoutError as exc:
            raise GatewayError(status_code=504, code="gateway_timeout", message="Model stream timed out") from exc
        except MessagesBackendError as exc:
            raise _safe_backend_error(error=exc) from exc
        except MessagesProviderError:
            raise
        except Exception as exc:
            logger.error("Host-only Messages stream failed (%s)", type(exc).__name__)
            raise GatewayError(
                status_code=502, code="backend_failed", message="Host-only Messages stream failed"
            ) from exc
        if not isinstance(stream, MessagesStream):
            raise GatewayError(
                status_code=502, code="invalid_backend_response", message="Model backend returned no Messages stream"
            )
        try:
            tracker = MessagesStreamValidator(
                validator=self._validator,
                advertised_tools=request.advertised_tools,
                max_tokens=request.max_tokens,
                max_bytes=self._limits.max_response_bytes,
            )
            self._validator.validate_provider_headers(headers=stream.headers, streaming=True)
            first = await self._next_frame_async(
                request=request, stream=stream, tracker=tracker, deadline=deadline, include_headers=True
            )
        except (GatewayError, asyncio.CancelledError):
            await self._close_stream_async(stream=stream)
            raise
        return StreamingResponse(
            self._stream_async(request=request, stream=stream, tracker=tracker, first=first, deadline=deadline),
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no", **dict(stream.headers)},
            media_type="text/event-stream",
        )

    async def _next_frame_async(
        self,
        *,
        request: MessagesRequest,
        stream: MessagesStream,
        tracker: MessagesStreamValidator,
        deadline: float,
        include_headers: bool = False,
    ) -> bytes:
        try:
            async with asyncio.timeout_at(deadline):
                frame = await anext(stream.frames)
        except StopAsyncIteration as exc:
            raise GatewayError(
                status_code=502, code="incomplete_stream", message="Anthropic stream ended without message_stop"
            ) from exc
        except TimeoutError as exc:
            raise GatewayError(status_code=504, code="gateway_timeout", message="Anthropic stream timed out") from exc
        except MessagesBackendError as exc:
            raise _safe_backend_error(error=exc) from exc
        except Exception as exc:
            logger.error("Host-only Messages stream failed (%s)", type(exc).__name__)
            raise GatewayError(
                status_code=502, code="backend_failed", message="Host-only Messages stream failed"
            ) from exc
        coverage = tracker.accept(frame=frame)
        await self._observe_async(
            request=request,
            kind=GatewayFrameKind.RESPONSE_EVENT,
            frame=frame,
            coverage=coverage,
            deadline=deadline,
            headers=stream.headers if include_headers else (),
        )
        return frame

    async def _stream_async(
        self,
        *,
        request: MessagesRequest,
        stream: MessagesStream,
        tracker: MessagesStreamValidator,
        first: bytes,
        deadline: float,
    ) -> AsyncGenerator[bytes, None]:
        try:
            yield first
            while not tracker.terminal:
                try:
                    frame = await self._next_frame_async(
                        request=request, stream=stream, tracker=tracker, deadline=deadline
                    )
                except GatewayError as error:
                    generated = _gateway_sse_error(error=error)
                    await self._record_gateway_error_async(
                        request=request, error=error, frame=generated, streaming=True
                    )
                    yield generated
                    return
                yield frame
        finally:
            await self._close_stream_async(stream=stream)

    async def _close_stream_async(self, *, stream: MessagesStream) -> None:
        try:
            await stream.close_async()
        except Exception as exc:
            logger.error("Could not close the Messages provider stream (%s)", type(exc).__name__)


def create_claude_messages_app(
    *,
    route: GatewayRoute,
    limits: GatewayLimits,
    backend: ModelOnlyMessagesBackend | None = None,
    observation_callback: MessagesObservationCallback | None = None,
) -> Starlette:
    """
    Create an isolated Claude Code Anthropic Messages endpoint, without a socket.

    ``claude -p`` sends Messages to ``ANTHROPIC_BASE_URL/v1/messages`` using
    ``ANTHROPIC_AUTH_TOKEN`` (an ephemeral run token) and a run-id header
    supplied through ``ANTHROPIC_CUSTOM_HEADERS``. The actual provider auth
    exists only in a separately injected host backend. The optional token
    counting, startup, and model-discovery endpoints are deliberately absent.

    Args:
        route (GatewayRoute): One run, exact upstream model ID, and guest-only bearer.
        limits (GatewayLimits): Positive run request/token/byte/time ceilings.
        backend (ModelOnlyMessagesBackend | None): Explicit model-only Messages implementation.
        observation_callback (MessagesObservationCallback | None): Original wire and host-error observer.

    Returns:
        Starlette: A strict ASGI Messages gateway to mount in one run.
    """
    gateway = _ClaudeMessagesGateway(
        route=route, limits=limits, backend=backend, observation_callback=observation_callback
    )
    app = Starlette(routes=[Route("/v1/messages", gateway._handle_async, methods=["POST"])])
    app.router.redirect_slashes = False
    return app
