# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Default-off authenticated original-evidence upload, not a browser execution API."""

from __future__ import annotations

import base64
import binascii
from typing import Annotated
from uuid import UUID  # noqa: TC003 (FastAPI resolves the runtime path annotation)

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import ValidationError

from pyrit.backend.services.original_evidence_admission import (
    OriginalEvidenceEnvelope,
    get_original_evidence_provider,
)
from pyrit.backend.services.original_evidence_service import OriginalEvidenceReceipt, get_original_evidence_service
from pyrit.backend.services.original_run_admission import OriginalAdmissionError
from pyrit.models.catalog.scenario import OriginalRunReason

router = APIRouter()


@router.post("/internal/original-evidence/{job_ref}", response_model=OriginalEvidenceReceipt)
async def intake_original_evidence_async(
    *,
    request: Request,
    job_ref: UUID,
    authorization: Annotated[str | None, Header()] = None,
    encoded_envelope: Annotated[str | None, Header(alias="X-PyRIT-Original-Envelope")] = None,
) -> OriginalEvidenceReceipt:
    """
    Admit one worker's exact binary archive and source-bound proof.

    Returns:
        OriginalEvidenceReceipt: Actual backend readback, not a worker-supplied score.
    """
    if get_original_evidence_provider() is None:
        raise HTTPException(status_code=503, detail=OriginalRunReason.RUNNER_NOT_CONFIGURED.value)
    auth_parts = (authorization or "").split()
    if len(auth_parts) != 2 or auth_parts[0].casefold() != "bearer" or not 32 <= len(auth_parts[1]) <= 2048:
        raise HTTPException(status_code=401, detail=OriginalRunReason.OPERATOR_NOT_AUTHORIZED.value)
    if encoded_envelope is None or len(encoded_envelope) > 16 * 1024:
        raise HTTPException(status_code=400, detail=OriginalRunReason.SOURCE_UNVERIFIED.value)
    try:
        raw = base64.b64decode(encoded_envelope, altchars=b"-_", validate=True)
        envelope = OriginalEvidenceEnvelope.model_validate_json(raw)
    except (ValueError, binascii.Error, ValidationError) as error:
        raise HTTPException(status_code=400, detail=OriginalRunReason.SOURCE_UNVERIFIED.value) from error
    if envelope.job_ref != job_ref or request.headers.get("content-encoding", "identity") != "identity":
        raise HTTPException(status_code=400, detail=OriginalRunReason.SOURCE_UNVERIFIED.value)
    if request.headers.get("content-type") != "application/octet-stream":
        raise HTTPException(status_code=415, detail=OriginalRunReason.SOURCE_UNVERIFIED.value)
    service = get_original_evidence_service()
    try:
        admission = await service.authorize_intake_async(capability=auth_parts[1], envelope=envelope)
        expected_bytes = envelope.archive_bytes or 0
        content_length = request.headers.get("content-length")
        if content_length is not None and (not content_length.isdecimal() or int(content_length) != expected_bytes):
            raise HTTPException(status_code=400, detail=OriginalRunReason.SOURCE_UNVERIFIED.value)
        archive = bytearray()
        async for chunk in request.stream():
            archive.extend(chunk)
            if len(archive) > expected_bytes or len(archive) > 16 * 1024 * 1024:
                raise HTTPException(status_code=413, detail=OriginalRunReason.SOURCE_UNVERIFIED.value)
        return await service.persist_async(admission=admission, archive=bytes(archive))
    except OriginalAdmissionError as error:
        status = (
            401
            if error.reason is OriginalRunReason.OPERATOR_NOT_AUTHORIZED
            else 409
            if error.reason in (OriginalRunReason.CAPACITY_BUSY, OriginalRunReason.ADMISSION_EXPIRED)
            else 503
            if error.reason is OriginalRunReason.RUNNER_NOT_CONFIGURED
            else 400
        )
        raise HTTPException(status_code=status, detail=error.reason.value) from error
