# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Additive manual multi-send endpoints; transcripts stay on the conversation API."""

import logging

from fastapi import APIRouter, HTTPException, status

from pyrit.backend.models.message_batches import MessageBatchRequest, MessageBatchStatus
from pyrit.backend.services.manual_send_scheduler import ManualSendConflictError, ManualSendQueueFullError
from pyrit.backend.services.multi_send_service import MessageBatchNotFoundError, get_multi_send_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/attacks", tags=["attacks"])


@router.post(
    "/{attack_result_id}/messages/batch",
    response_model=MessageBatchStatus,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_message_batch_async(*, attack_result_id: str, request: MessageBatchRequest) -> MessageBatchStatus:
    """
    Accept a user-message batch and return a transient handle before expensive preparation.

    Returns:
        MessageBatchStatus: Compact live progress, not full transcripts.
    """
    try:
        return await get_multi_send_service().submit_async(attack_result_id=attack_result_id, request=request)
    except MessageBatchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ManualSendConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ManualSendQueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to accept message batch for attack '%s'", attack_result_id)
        raise HTTPException(status_code=500, detail="Could not accept the message batch. Check server logs.") from exc


@router.get("/{attack_result_id}/message-batches/{batch_id}", response_model=MessageBatchStatus)
async def get_message_batch_async(*, attack_result_id: str, batch_id: str) -> MessageBatchStatus:
    """
    Read a live batch handle; missing or expired handles never trigger replay.

    Returns:
        MessageBatchStatus: Compact progress for the requested attack only.
    """
    try:
        return get_multi_send_service().get_status(attack_result_id=attack_result_id, batch_id=batch_id)
    except MessageBatchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
