# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Default-off authenticated evaluation job port, independent of ordinary Scenario execution."""

from __future__ import annotations

from typing import Annotated, TypeVar
from uuid import UUID  # noqa: TC003 (FastAPI resolves path annotations)

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, ValidationError
from starlette.responses import Response

from pyrit.backend.middleware.auth import get_authenticated_operator
from pyrit.backend.models.evaluation_jobs import EvaluationCaseLaunch, EvaluationOriginalResult
from pyrit.backend.services.evaluation_job_application import EvaluationJobApplication
from pyrit.executor.jobs.local import LocalEvaluationJobPort
from pyrit.executor.jobs.port import EvaluationJobError, EvaluationJobErrorCode
from pyrit.models.evaluation_job import (
    EvaluationControlReceipt,
    EvaluationControlRequest,
    EvaluationJobRegistration,
    EvaluationJobRequest,
    EvaluationJobSnapshot,
    EvaluationJobSubmission,
)

router = APIRouter()
T = TypeVar("T", bound=BaseModel)


def _admission(request: Request) -> tuple[LocalEvaluationJobPort, str]:
    port = getattr(request.app.state, "evaluation_job_port", None)
    if not isinstance(port, LocalEvaluationJobPort):
        raise HTTPException(status_code=503, detail="evaluation_job_backend_not_configured")
    user = get_authenticated_operator(request)
    if user is None:
        raise HTTPException(status_code=401, detail=EvaluationJobErrorCode.NOT_AUTHORIZED.value)
    application = getattr(request.app.state, "evaluation_job_application", None)
    if isinstance(application, EvaluationJobApplication):
        try:
            application.authorize_actor(user.oid)
        except EvaluationJobError as error:
            raise _http_error(error) from error
    return port, user.oid


def _application(request: Request) -> EvaluationJobApplication:
    application = getattr(request.app.state, "evaluation_job_application", None)
    if not isinstance(application, EvaluationJobApplication):
        raise HTTPException(status_code=503, detail="evaluation_job_application_not_configured")
    return application


def _http_error(error: EvaluationJobError) -> HTTPException:
    status = {
        EvaluationJobErrorCode.NOT_AUTHORIZED: 403,
        EvaluationJobErrorCode.NOT_FOUND: 404,
        EvaluationJobErrorCode.CLOSED: 503,
        EvaluationJobErrorCode.UNSUPPORTED_RUNTIME: 400,
        EvaluationJobErrorCode.UNSUPPORTED_CONTROL: 400,
    }.get(error.code, 409)
    return HTTPException(status_code=status, detail=error.code.value)


async def _message_async(*, request: Request, model: type[T]) -> T:
    if request.headers.get("content-type", "").split(";")[0] != "application/json":
        raise HTTPException(status_code=415, detail="evaluation_job_requires_json")
    if request.headers.get("content-encoding", "identity") != "identity":
        raise HTTPException(status_code=400, detail="evaluation_job_invalid_encoding")
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > 32 * 1024:
            raise HTTPException(status_code=413, detail="evaluation_job_message_too_large")
    try:
        return model.model_validate_json(bytes(content))
    except ValidationError as error:
        raise HTTPException(status_code=400, detail="evaluation_job_invalid_message") from error


@router.get("/evaluation-jobs/catalog", response_model=tuple[EvaluationJobRegistration, ...])
async def catalog_async(*, request: Request) -> tuple[EvaluationJobRegistration, ...]:
    """
    List only installed capabilities, not universal runtime support.

    Returns:
        tuple[EvaluationJobRegistration, ...]: The sole harmless binding in this local PoC.
    """
    port, actor = _admission(request)
    if isinstance(getattr(request.app.state, "evaluation_job_application", None), EvaluationJobApplication):
        return ()
    try:
        return await port.catalog_async(actor_id=actor)
    except EvaluationJobError as error:
        raise _http_error(error) from error


@router.post("/evaluation-jobs", response_model=EvaluationJobSubmission, status_code=202)
async def submit_async(*, request: Request) -> EvaluationJobSubmission:
    """
    Persist authenticated immutable admission without claiming execution or a grade.

    Returns:
        EvaluationJobSubmission: Durable queue acknowledgment.
    """
    port, actor = _admission(request)
    if isinstance(getattr(request.app.state, "evaluation_job_application", None), EvaluationJobApplication):
        raise HTTPException(status_code=403, detail=EvaluationJobErrorCode.NOT_AUTHORIZED.value)
    message = await _message_async(request=request, model=EvaluationJobRequest)
    try:
        return await port.submit_async(request=message, actor_id=actor)
    except EvaluationJobError as error:
        raise _http_error(error) from error


@router.post("/evaluation-jobs/cases/{case_ref}", response_model=EvaluationJobSubmission, status_code=202)
async def launch_case_async(*, request: Request, case_ref: str) -> EvaluationJobSubmission:
    """
    Launch only an installed opaque logical case with an actor-bound idempotency key.

    Returns:
        EvaluationJobSubmission: Existing durable queue acknowledgment, not execution or a grade.
    """
    _, actor = _admission(request)
    application = _application(request)
    if request.query_params:
        raise HTTPException(status_code=400, detail="evaluation_job_invalid_message")
    launch = await _message_async(request=request, model=EvaluationCaseLaunch)
    try:
        return await application.submit_async(case_ref=case_ref, launch=launch, actor_id=actor)
    except EvaluationJobError as error:
        raise _http_error(error) from error


@router.get("/evaluation-jobs/{job_id}", response_model=EvaluationJobSnapshot)
async def status_async(
    *, request: Request, job_id: UUID, after_sequence: Annotated[int, Query(ge=0)] = 0
) -> EvaluationJobSnapshot:
    """
    Return an actor-bound ordered page without raw worker data or secrets.

    Returns:
        EvaluationJobSnapshot: Source/canonical/cleanup state independently of queue delivery.
    """
    port, actor = _admission(request)
    try:
        return await port.status_async(job_id=job_id, actor_id=actor, after_sequence=after_sequence)
    except EvaluationJobError as error:
        raise _http_error(error) from error


@router.post("/evaluation-jobs/{job_id}/cancel", response_model=EvaluationJobSnapshot)
async def cancel_async(*, request: Request, job_id: UUID) -> EvaluationJobSnapshot:
    """
    Request owned cancellation; irreversible canonical publication is not rolled back.

    Returns:
        EvaluationJobSnapshot: Actual cancellation request or terminal state.
    """
    port, actor = _admission(request)
    try:
        return await port.cancel_async(job_id=job_id, actor_id=actor)
    except EvaluationJobError as error:
        raise _http_error(error) from error


@router.get("/evaluation-jobs/{job_id}/result", response_model=EvaluationOriginalResult)
async def result_async(*, request: Request, response: Response, job_id: UUID) -> EvaluationOriginalResult:
    """
    Reverify API-owned original evidence and linked rows before returning an authorized grade.

    Returns:
        EvaluationOriginalResult: Separate original grade, PyRIT outcome, custody and cleanup state.
    """
    _, actor = _admission(request)
    response.headers["Cache-Control"] = "no-store"
    try:
        return await _application(request).result_async(job_id=job_id, actor_id=actor)
    except EvaluationJobError as error:
        raise _http_error(error) from error


@router.get("/evaluation-jobs/{job_id}/archive", response_class=Response)
async def archive_async(*, request: Request, job_id: UUID) -> Response:
    """
    Download only the authorized job's exact bounded original archive from canonical API storage.

    Returns:
        Response: Retained unchanged binary `.eval`, never a worker path or synthesized log.
    """
    _, actor = _admission(request)
    try:
        snapshot, retained = await _application(request).readback_async(job_id=job_id, actor_id=actor)
    except EvaluationJobError as error:
        raise _http_error(error) from error
    assert snapshot.canonical is not None
    return Response(
        content=retained.archive,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": 'attachment; filename="original.eval"',
            "ETag": f'"{snapshot.canonical.artifact_sha256}"',
        },
    )


@router.post("/evaluation-jobs/{job_id}/control", response_model=EvaluationControlReceipt)
async def control_async(
    *,
    request: Request,
    job_id: UUID,
    capability: Annotated[str | None, Header(alias="X-PyRIT-Job-Control")] = None,
) -> EvaluationControlReceipt:
    """
    Admit structured data only at an active reviewed capability boundary.

    Returns:
        EvaluationControlReceipt: Command acceptance, not proof of agent execution.
    """
    port, actor = _admission(request)
    if capability is None:
        raise HTTPException(status_code=403, detail=EvaluationJobErrorCode.NOT_AUTHORIZED.value)
    command = await _message_async(request=request, model=EvaluationControlRequest)
    try:
        return await port.control_async(job_id=job_id, actor_id=actor, capability=capability, command=command)
    except EvaluationJobError as error:
        raise _http_error(error) from error
