# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Loopback-compatible native model relay; never a browser-selectable provider."""

import asyncio
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, Response

from pyrit.backend.services.original_model_relay import OriginalModelRelay, OriginalRelayError
from pyrit.backend.services.original_worker_runtime import OriginalWorkerRuntime

router = APIRouter()


def _relay(request: Request) -> OriginalModelRelay:
    runtime = getattr(request.app.state, "original_worker_runtime", None)
    if not isinstance(runtime, OriginalWorkerRuntime):
        raise HTTPException(status_code=503, detail="Original evaluated relay is not configured.")
    return runtime.relay


def _capability(request: Request) -> str:
    parts = request.headers.get("Authorization", "").split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not 43 <= len(parts[1]) <= 128:
        raise HTTPException(status_code=403, detail="Original evaluated capability is invalid.")
    return parts[1]


async def _read_body_async(
    *,
    request: Request,
    relay: OriginalModelRelay,
    job_ref: UUID,
    limit: int,
    kind: Literal["request", "close"],
) -> bytes:
    """
    Bound authenticated body reading independently of owned upstream work.

    Returns:
        bytes: The exact bounded authenticated body.
    """
    body = bytearray()
    try:
        async with asyncio.timeout(relay.REQUEST_BODY_TIMEOUT_SECONDS):
            async for chunk in request.stream():
                if len(chunk) > limit - len(body):
                    relay.revoke(job_ref=job_ref)
                    raise OriginalRelayError(code=f"relay_{kind}_body_too_large", status_code=413)
                body.extend(chunk)
    except TimeoutError:
        relay.revoke(job_ref=job_ref)
        raise OriginalRelayError(code=f"relay_{kind}_body_timeout", status_code=408) from None
    except ClientDisconnect:
        relay.revoke(job_ref=job_ref)
        raise OriginalRelayError(code=f"relay_{kind}_body_disconnected", status_code=400) from None
    return bytes(body)


@router.post("/internal/original-model/{job_ref}/chat/completions", include_in_schema=False)
async def original_model_completion_async(*, job_ref: UUID, request: Request) -> Response:
    """
    Authenticate before reading and cap the request before native credential/provider use.

    Returns:
        Response: Exact bounded upstream JSON or a finite safe refusal.
    """
    relay, capability = _relay(request), _capability(request)
    try:
        relay.authenticate(job_ref=job_ref, capability=capability)
        body = await _read_body_async(
            request=request, relay=relay, job_ref=job_ref, limit=relay.config.max_request_bytes, kind="request"
        )
        kind = request.headers.get("X-PyRIT-Original-Request-Kind", "original")
        if kind not in ("original", "qualification"):
            relay.refuse(job_ref=job_ref, capability=capability, code="relay_request_kind_invalid", status_code=400)
        request_kind: Literal["original", "qualification"] = "qualification" if kind == "qualification" else "original"
        content, request_id = await relay.forward_async(
            job_ref=job_ref,
            capability=capability,
            content=body,
            request_kind=request_kind,
            inspect_request_id=request.headers.get("x-irid"),
        )
        return Response(
            content=content,
            media_type="application/json",
            headers={"X-PyRIT-Original-Request-ID": request_id, "Cache-Control": "no-store"},
        )
    except OriginalRelayError as error:
        return JSONResponse({"detail": error.code}, status_code=error.status_code)


@router.post("/internal/original-model/{job_ref}/close", include_in_schema=False)
async def original_model_close_async(*, job_ref: UUID, request: Request) -> Response:
    """
    Revoke and drain only this exact role; accepted during owned shutdown.

    Returns:
        Response: Actual native inflight/usage closure, never a guessed stopped state.
    """
    relay, capability = _relay(request), _capability(request)
    try:
        relay.authenticate(job_ref=job_ref, capability=capability, allow_closed=True)
        relay.revoke(job_ref=job_ref)
        await _read_body_async(request=request, relay=relay, job_ref=job_ref, limit=1024, kind="close")
        return JSONResponse(
            await relay.close_async(job_ref=job_ref, capability=capability),
            headers={"Cache-Control": "no-store"},
        )
    except OriginalRelayError as error:
        return JSONResponse({"detail": error.code}, status_code=error.status_code)
