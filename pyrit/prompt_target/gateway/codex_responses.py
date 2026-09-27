# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run-scoped, model-only Responses ASGI endpoint for a sandboxed Codex CLI."""

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

from pyrit.prompt_target.gateway.responses_contract import (
    BackendCapabilities,
    GatewayCoverage,
    GatewayError,
    GatewayFrameKind,
    GatewayLimits,
    GatewayObservation,
    GatewayRoute,
    ModelBackendError,
    ModelBackendErrorCode,
    ModelOnlyResponsesBackend,
    ModelRequest,
    ObservationCallback,
)
from pyrit.prompt_target.gateway.responses_validation import (
    ResponsesStreamValidator,
    ResponsesValidator,
    strict_json_loads,
)

logger = logging.getLogger(__name__)


def _error_body(*, error: GatewayError) -> dict[str, dict[str, str | None]]:
    if error.status_code in {401, 403}:
        error_type = "authentication_error"
    elif error.status_code == 429:
        error_type = "rate_limit_error"
    elif error.status_code < 500:
        error_type = "invalid_request_error"
    else:
        error_type = "api_error"
    return {
        "error": {
            "message": str(error),
            "type": error_type,
            "code": error.code,
            "param": error.param,
        }
    }


def _json_error(*, error: GatewayError) -> JSONResponse:
    headers = {"Cache-Control": "no-store"}
    if error.status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(_error_body(error=error), status_code=error.status_code, headers=headers)


def _sse_error(*, error: GatewayError) -> bytes:
    data = json.dumps(_error_body(error=error), separators=(",", ":"))
    return f"event: error\ndata: {data}\n\n".encode()


def _backend_gateway_error(*, error: ModelBackendError) -> GatewayError:
    return GatewayError(status_code=error.status_code, code=error.code.value, message=str(error))


class _CodexResponsesGateway:
    """Own a single run's authorization, model route, and conservative budgets."""

    _BACKEND_ERROR_CODES = frozenset(code.value for code in ModelBackendErrorCode)

    def __init__(
        self,
        *,
        route: GatewayRoute,
        limits: GatewayLimits,
        backend: ModelOnlyResponsesBackend | None,
        observation_callback: ObservationCallback | None,
    ) -> None:
        self._route = route
        self._limits = limits
        self._backend = backend
        self._observation_callback = observation_callback
        self._validator = ResponsesValidator(
            route=route,
            limits=limits,
            capabilities=backend.capabilities if backend else BackendCapabilities(),
        )
        self._lock = asyncio.Lock()
        self._requests = 0
        self._reserved_request_bytes = 0
        self._reserved_output_tokens = 0

    async def _handle_async(self, request: Request) -> Response:
        deadline = asyncio.get_running_loop().time() + self._limits.timeout_seconds
        model_request: ModelRequest | None = None
        streaming = False
        try:
            self._authenticate(request=request)
            if request.scope.get("query_string"):
                raise GatewayError(
                    status_code=501, code="unsupported_feature", message="Responses query parameters are not supported"
                )
            raw_request = await self._read_body_async(request=request, deadline=deadline)
            body = self._decode_body(raw=raw_request)
            validated = self._validator.validate_request(body=body)
            streaming = validated.streaming
            if self._backend is None:
                raise GatewayError(
                    status_code=501,
                    code="model_backend_required",
                    message=(
                        "No model-only Responses backend is configured; OpenAIResponseTarget "
                        "does not provide Responses SSE or a CLI-owned tool loop"
                    ),
                )
            await self._reserve_async(tokens=validated.output_token_limit, request_bytes=len(raw_request))
            model_request = ModelRequest(
                run_id=self._route.run_id,
                request_id=uuid4().hex,
                body=validated.body,
                output_token_limit=validated.output_token_limit,
            )
            await self._observe_async(
                request=model_request,
                kind=GatewayFrameKind.REQUEST,
                frame=raw_request,
                coverage=validated.coverage,
                deadline=deadline,
            )
            if validated.streaming:
                return await self._start_stream_async(request=model_request, deadline=deadline)
            return await self._complete_async(request=model_request, deadline=deadline)
        except GatewayError as error:
            reply = _json_error(error=error)
            if model_request is not None and error.code in self._BACKEND_ERROR_CODES:
                await self._observe_gateway_error_async(
                    request=model_request, error=error, frame=bytes(reply.body), streaming=streaming
                )
            return reply

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
        run_ids = request.headers.getlist("x-pyrit-run-id")
        if len(run_ids) != 1 or run_ids[0] != self._route.run_id:
            raise GatewayError(status_code=403, code="invalid_run", message="Request is not routed to this run")

    async def _read_body_async(self, *, request: Request, deadline: float) -> bytes:
        types = request.headers.getlist("content-type")
        if len(types) != 1 or types[0].split(";", maxsplit=1)[0].strip().lower() != "application/json":
            raise GatewayError(
                status_code=415, code="invalid_content_type", message="Content-Type must be application/json"
            )
        lengths = request.headers.getlist("content-length")
        if len(lengths) > 1 or (
            lengths and (len(lengths[0]) > 20 or not lengths[0].isascii() or not lengths[0].isdigit())
        ):
            raise GatewayError(status_code=400, code="invalid_content_length", message="Invalid Content-Length")
        if lengths and int(lengths[0]) > self._limits.max_request_bytes:
            raise GatewayError(
                status_code=413, code="request_too_large", message="Responses request exceeded byte limit"
            )
        collected = bytearray()
        try:
            async with asyncio.timeout_at(deadline):
                async for chunk in request.stream():
                    if len(collected) + len(chunk) > self._limits.max_request_bytes:
                        raise GatewayError(
                            status_code=413, code="request_too_large", message="Responses request exceeded byte limit"
                        )
                    collected.extend(chunk)
        except TimeoutError as exc:
            raise GatewayError(status_code=504, code="gateway_timeout", message="Responses request timed out") from exc
        return bytes(collected)

    def _decode_body(self, *, raw: bytes) -> dict[str, Any]:
        try:
            body = strict_json_loads(value=raw)
        except (ValueError, RecursionError) as exc:
            raise GatewayError(
                status_code=400, code="invalid_json", message="Responses body must be valid JSON"
            ) from exc
        if not isinstance(body, dict) or not all(isinstance(key, str) for key in body):
            raise GatewayError(status_code=400, code="invalid_json", message="Responses body must be a JSON object")
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

    async def _observe_async(
        self,
        *,
        request: ModelRequest,
        kind: GatewayFrameKind,
        frame: bytes,
        coverage: frozenset[GatewayCoverage],
        deadline: float,
        error: GatewayError | None = None,
    ) -> None:
        if self._observation_callback is None:
            return
        observation = GatewayObservation(
            run_id=request.run_id,
            request_id=request.request_id,
            kind=kind,
            frame=frame,
            coverage=coverage,
            error_code=error.code if error is not None else None,
            status_code=error.status_code if error is not None else None,
        )
        try:
            async with asyncio.timeout_at(deadline):
                await self._observation_callback(observation)
        except TimeoutError as exc:
            raise GatewayError(
                status_code=504, code="gateway_timeout", message="Gateway observation timed out"
            ) from exc
        except Exception as exc:
            logger.error("Gateway observation failed (%s)", type(exc).__name__)
            raise GatewayError(
                status_code=500, code="observation_failed", message="Host observation callback failed"
            ) from exc

    async def _observe_gateway_error_async(
        self, *, request: ModelRequest, error: GatewayError, frame: bytes, streaming: bool
    ) -> None:
        coverage = {GatewayCoverage.FAILED}
        if streaming:
            coverage.add(GatewayCoverage.STREAMING)
        try:
            await self._observe_async(
                request=request,
                kind=GatewayFrameKind.GATEWAY_ERROR,
                frame=frame,
                coverage=frozenset(coverage),
                deadline=asyncio.get_running_loop().time() + min(1.0, self._limits.timeout_seconds),
                error=error,
            )
        except GatewayError as observation_error:
            logger.error(
                "Could not record stream failure %s (%s)",
                error.code,
                observation_error.code,
            )

    async def _report_stream_error_async(self, *, request: ModelRequest, error: GatewayError) -> bytes:
        frame = _sse_error(error=error)
        await self._observe_gateway_error_async(request=request, error=error, frame=frame, streaming=True)
        return frame

    async def _complete_async(self, *, request: ModelRequest, deadline: float) -> Response:
        if self._backend is None:
            raise GatewayError(status_code=501, code="model_backend_required", message="Model backend is unavailable")
        try:
            async with asyncio.timeout_at(deadline):
                raw_response = await self._backend.create_response_async(request=request)
        except TimeoutError as exc:
            raise GatewayError(status_code=504, code="gateway_timeout", message="Model response timed out") from exc
        except ModelBackendError as exc:
            raise _backend_gateway_error(error=exc) from exc
        except NotImplementedError as exc:
            raise GatewayError(
                status_code=501, code="backend_unsupported", message="Model backend cannot create Responses"
            ) from exc
        except Exception as exc:
            logger.error("Model-only Responses backend failed (%s)", type(exc).__name__)
            raise GatewayError(
                status_code=502, code="backend_failed", message="Model-only Responses backend failed"
            ) from exc
        if not isinstance(raw_response, bytes):
            raise GatewayError(
                status_code=502,
                code="invalid_backend_response",
                message="Model backend must return Responses JSON bytes",
            )
        if len(raw_response) > self._limits.max_response_bytes:
            raise GatewayError(status_code=502, code="response_too_large", message="Model response exceeded byte limit")
        coverage = self._validator.validate_response(frame=raw_response, request=request)
        await self._observe_async(
            request=request,
            kind=GatewayFrameKind.RESPONSE,
            frame=raw_response,
            coverage=coverage,
            deadline=deadline,
        )
        return Response(content=raw_response, media_type="application/json", headers={"Cache-Control": "no-store"})

    async def _start_stream_async(self, *, request: ModelRequest, deadline: float) -> Response:
        if self._backend is None:
            raise GatewayError(status_code=501, code="model_backend_required", message="Model backend is unavailable")
        try:
            stream = self._backend.stream_response_async(request=request)
        except ModelBackendError as exc:
            raise _backend_gateway_error(error=exc) from exc
        except NotImplementedError as exc:
            raise GatewayError(
                status_code=501, code="backend_unsupported", message="Model backend cannot stream Responses SSE"
            ) from exc
        except Exception as exc:
            logger.error("Model-only Responses stream could not start (%s)", type(exc).__name__)
            raise GatewayError(
                status_code=502, code="backend_failed", message="Model-only Responses stream failed"
            ) from exc
        if not isinstance(stream, AsyncGenerator):
            raise GatewayError(
                status_code=502,
                code="invalid_backend_response",
                message="Model backend must return an async generator of complete SSE frames",
            )
        tracker = ResponsesStreamValidator(
            validator=self._validator, request=request, max_bytes=self._limits.max_response_bytes
        )
        try:
            first = await self._next_frame_async(stream=stream, tracker=tracker, request=request, deadline=deadline)
        except (GatewayError, asyncio.CancelledError):
            await stream.aclose()
            raise
        return StreamingResponse(
            self._stream_frames_async(first=first, stream=stream, tracker=tracker, request=request, deadline=deadline),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no", "X-Content-Type-Options": "nosniff"},
        )

    async def _next_frame_async(
        self,
        *,
        stream: AsyncGenerator[bytes, None],
        tracker: ResponsesStreamValidator,
        request: ModelRequest,
        deadline: float,
    ) -> bytes:
        try:
            async with asyncio.timeout_at(deadline):
                frame = await anext(stream)
        except StopAsyncIteration as exc:
            raise GatewayError(
                status_code=502, code="incomplete_stream", message="Responses stream ended without [DONE]"
            ) from exc
        except TimeoutError as exc:
            raise GatewayError(status_code=504, code="gateway_timeout", message="Responses stream timed out") from exc
        except ModelBackendError as exc:
            raise _backend_gateway_error(error=exc) from exc
        except NotImplementedError as exc:
            raise GatewayError(
                status_code=501, code="backend_unsupported", message="Model backend cannot stream Responses SSE"
            ) from exc
        except Exception as exc:
            logger.error("Model-only Responses stream failed (%s)", type(exc).__name__)
            raise GatewayError(
                status_code=502, code="backend_failed", message="Model-only Responses stream failed"
            ) from exc
        coverage = tracker.accept(frame=frame)
        await self._observe_async(
            request=request,
            kind=GatewayFrameKind.RESPONSE_EVENT,
            frame=frame,
            coverage=coverage,
            deadline=deadline,
        )
        return frame

    async def _stream_frames_async(
        self,
        *,
        first: bytes,
        stream: AsyncGenerator[bytes, None],
        tracker: ResponsesStreamValidator,
        request: ModelRequest,
        deadline: float,
    ) -> AsyncGenerator[bytes, None]:
        try:
            yield first
            while not tracker.done:
                try:
                    frame = await self._next_frame_async(
                        stream=stream, tracker=tracker, request=request, deadline=deadline
                    )
                except GatewayError as error:
                    yield await self._report_stream_error_async(request=request, error=error)
                    return
                yield frame
        finally:
            await stream.aclose()


def create_codex_responses_app(
    *,
    route: GatewayRoute,
    limits: GatewayLimits,
    backend: ModelOnlyResponsesBackend | None = None,
    observation_callback: ObservationCallback | None = None,
) -> Starlette:
    """
    Create a single-run Codex Responses endpoint without opening a socket.

    The sandbox gets only ``route.guest_token`` and its run id, never a primary
    provider credential. Configure Codex's custom provider ``base_url`` to
    end in ``/v1``, set ``wire_api = "responses"``, and supply the run id via
    the ``X-PyRIT-Run-ID`` header. Only ``POST /v1/responses`` is mounted.
    No provider target or host tool executor is created by this gateway.

    Args:
        route (GatewayRoute): Host-generated run identity, model alias, and ephemeral guest token.
        limits (GatewayLimits): Nonzero request, token, time, and byte ceilings.
        backend (ModelOnlyResponsesBackend | None): Trusted model-only backend. Without one, requests fail with 501.
        observation_callback (ObservationCallback | None): Host callback for original request/response wire frames.

    Returns:
        Starlette: An ASGI application to mount only inside an isolated run.
    """
    gateway = _CodexResponsesGateway(
        route=route, limits=limits, backend=backend, observation_callback=observation_callback
    )
    app = Starlette(routes=[Route("/v1/responses", gateway._handle_async, methods=["POST"])])
    app.router.redirect_slashes = False
    return app
