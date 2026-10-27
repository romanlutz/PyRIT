# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Capture ordering around an already-admitted host forwarder, without a client or SDK shim."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Protocol

from pyrit.models.native_harness import NativeModelCaptureIdentity, NativeModelResponseOrigin


class NativeModelObserver(Protocol):
    """One exact turn's durable body observer; no dispatch, credentials or grading."""

    @property
    def identity(self) -> NativeModelCaptureIdentity:
        """The immutable capture identity."""
        ...

    @property
    def max_body_bytes(self) -> int:
        """The store's original-body size limit."""
        ...

    def ensure_active(self) -> None:
        """Reject a closed, mismatched or failed capture turn."""
        ...

    def fail(self, reason: str) -> None:
        """Latch a content-free failure for durable controller checkpointing."""
        ...

    async def record_request_async(self, *, request_id: str, body: bytes, accepted: bool) -> None:
        """Persist the original request before forwarding."""
        ...

    async def record_response_async(
        self, *, request_id: str, body: bytes, status_code: int, origin: NativeModelResponseOrigin, complete: bool
    ) -> None:
        """Persist original response bytes and explicit provenance before return."""
        ...


@dataclass(frozen=True, kw_only=True)
class NativeModelRequest:
    """Caller-admitted original body and identity; no routing, auth headers or credentials."""

    identity: NativeModelCaptureIdentity
    request_id: str
    body: bytes = field(repr=False)
    streaming: bool


@dataclass(frozen=True, kw_only=True)
class NativeModelResponse:
    """Explicit trusted origin, original body and caller-required metadata, never rewritten."""

    identity: NativeModelCaptureIdentity
    request_id: str
    body: bytes = field(repr=False)
    status_code: int
    origin: NativeModelResponseOrigin
    complete: bool
    headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)


class NativeModelForwarder(Protocol):
    """
    Forward once on one already-admitted host route, with no implicit retry or tool execution.

    The caller owns protocol validation, route, auth, admission, run budgets and safe
    response headers/exceptions. Honor the same absolute event-loop deadline and
    bound response acquisition by ``max_response_bytes``. Return original whole
    bodies with an explicit origin, not inferred from status or content. Cancel
    and join any owned I/O/workers before returning or raising on cancellation.
    This is not a Copilot SDK request-handler interface or a streaming protocol.
    """

    async def forward_async(
        self, *, request: NativeModelRequest, deadline: float, max_response_bytes: int
    ) -> NativeModelResponse:
        """Return one original response, or raise a credential-free transport exception."""
        ...


@dataclass(frozen=True, kw_only=True)
class NativeModelTransport:
    """Enforce evidence boundaries; never select routes, authorize, normalize bodies, grade or publish."""

    observer: NativeModelObserver = field(repr=False)
    forwarder: NativeModelForwarder = field(repr=False)

    @property
    def identity(self) -> NativeModelCaptureIdentity:
        """The immutable run/session/turn identity of this transport instance."""
        return self.observer.identity

    async def forward_async(self, *, request: NativeModelRequest, deadline: float) -> NativeModelResponse:
        """
        Persist the request, forward once, then persist the response before returning it.

        Args:
            request (NativeModelRequest): Original, already-admitted non-streaming body.
            deadline (float): Caller-owned absolute event-loop time for capture and forwarding.

        Returns:
            NativeModelResponse: The identical forwarder result, including origin and headers.

        Raises:
            ValueError: If identity, deadline, protocol or response metadata is invalid.
            RuntimeError: If source capture failed or the response is incomplete.
            TimeoutError: If the caller's deadline is exhausted.
            asyncio.CancelledError: After cancelling and joining the local forwarder task.
            Exception: The original forwarder or durable capture exception, without retry.
        """
        try:
            self._check_boundary(request=request, deadline=deadline)
            if request.streaming is not False:
                raise ValueError("The observation transport supports only explicit non-streaming whole bodies.")
            async with asyncio.timeout_at(deadline):
                await self.observer.record_request_async(
                    request_id=request.request_id, body=request.body, accepted=True
                )
                self._check_boundary(request=request, deadline=deadline)
                response = await self._forward_once_async(request=request, deadline=deadline)
                await self._capture_response_async(request=request, response=response, deadline=deadline)
                return response
        except (Exception, asyncio.CancelledError):
            self.observer.fail("Trusted model forwarding or its durable capture did not complete.")
            raise

    async def record_rejection_async(
        self, *, request: NativeModelRequest, response: NativeModelResponse, deadline: float
    ) -> NativeModelResponse:
        """
        Retain a caller's explicit rejection without invoking the forwarder or inventing a body.

        Returns:
            NativeModelResponse: The identical, non-provider rejection after durable capture.

        Raises:
            ValueError: If rejection identity, origin, metadata or status is invalid.
            RuntimeError: If capture failed or the rejection body is incomplete.
            TimeoutError: If the same caller-owned capture deadline is exhausted.
            asyncio.CancelledError: If capture is cancelled; in-flight writes are joined.
            Exception: The original durable capture exception.
        """
        try:
            self._check_boundary(request=request, deadline=deadline)
            self._validate_response(request=request, response=response)
            if (
                response.origin not in {NativeModelResponseOrigin.HOST, NativeModelResponseOrigin.SYNTHETIC}
                or type(response.status_code) is not int
                or not 400 <= response.status_code <= 599
            ):
                raise ValueError("An explicit rejection requires a non-provider error response.")
            async with asyncio.timeout_at(deadline):
                await self.observer.record_request_async(
                    request_id=request.request_id, body=request.body, accepted=False
                )
                await self._capture_response_async(request=request, response=response, deadline=deadline)
                return response
        except (Exception, asyncio.CancelledError):
            self.observer.fail("Capturing an explicit model rejection did not complete.")
            raise

    def _check_boundary(self, *, request: NativeModelRequest, deadline: float) -> None:
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            raise ValueError("Model observation requires a finite absolute caller deadline.")
        if not isinstance(request, NativeModelRequest) or request.identity != self.identity:
            raise ValueError("Model request identity does not match the bound observer.")
        if type(request.streaming) is not bool:
            raise ValueError("Model request streaming mode must be explicit.")
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise asyncio.CancelledError
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("The caller's model observation deadline expired.")
        self.observer.ensure_active()

    def _validate_response(self, *, request: NativeModelRequest, response: NativeModelResponse) -> None:
        if (
            not isinstance(response, NativeModelResponse)
            or response.identity != self.identity
            or response.request_id != request.request_id
        ):
            raise ValueError("Model response identity does not match the observed request.")
        if type(response.headers) is not tuple or any(
            type(pair) is not tuple or len(pair) != 2 or any(type(value) is not str for value in pair)
            for pair in response.headers
        ):
            raise ValueError("Model response headers must be immutable name/value pairs.")

    async def _capture_response_async(
        self, *, request: NativeModelRequest, response: NativeModelResponse, deadline: float
    ) -> None:
        self._check_boundary(request=request, deadline=deadline)
        self._validate_response(request=request, response=response)
        await self.observer.record_response_async(
            request_id=request.request_id,
            body=response.body,
            status_code=response.status_code,
            origin=response.origin,
            complete=response.complete,
        )
        self._check_boundary(request=request, deadline=deadline)
        if not response.complete:
            raise RuntimeError("The trusted forwarder returned an incomplete model body.")

    async def _invoke_forwarder_async(self, *, request: NativeModelRequest, deadline: float) -> NativeModelResponse:
        self._check_boundary(request=request, deadline=deadline)
        return await self.forwarder.forward_async(
            request=request, deadline=deadline, max_response_bytes=self.observer.max_body_bytes
        )

    async def _forward_once_async(self, *, request: NativeModelRequest, deadline: float) -> NativeModelResponse:
        task = asyncio.create_task(self._invoke_forwarder_async(request=request, deadline=deadline))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            self.observer.fail("Model forwarding was cancelled before a durably captured response.")
            if not task.done():
                task.cancel()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception as error:
                    raise cancellation from error
            if not task.cancelled():
                try:
                    task.result()
                except Exception as error:
                    raise cancellation from error
            raise
