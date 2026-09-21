# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Submit and observe manual sends, including ordinary sends with count one."""

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from pyrit.backend.models.message_sends import MessageSendRequest, MessageSendStatus
from pyrit.backend.services.manual_send_scheduler import ManualSendConflictError, ManualSendQueueFullError
from pyrit.backend.services.message_send_service import MessageSendNotFoundError, get_message_send_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/attacks", tags=["attacks"])


@router.post(
    "/{attack_result_id}/message-sends",
    response_model=MessageSendStatus,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_message_send_async(*, attack_result_id: str, request: MessageSendRequest) -> MessageSendStatus:
    """
    Accept a send operation and return a transient handle before expensive preparation.

    Returns:
        MessageSendStatus: Compact live progress, not full transcripts.
    """
    try:
        return await get_message_send_service().submit_async(attack_result_id=attack_result_id, request=request)
    except MessageSendNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ManualSendConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ManualSendQueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to accept message send for attack '%s'", attack_result_id)
        raise HTTPException(status_code=500, detail="Could not accept the message send. Check server logs.") from exc


@router.get("/{attack_result_id}/message-sends/{send_id}", response_model=MessageSendStatus)
async def get_message_send_async(
    *, attack_result_id: str, send_id: str, wait_ms: Annotated[int, Query(ge=0, le=1000)] = 0
) -> MessageSendStatus:
    """
    Read progress, optionally waiting briefly for completion without resubmitting.

    Returns:
        MessageSendStatus: Compact progress for the requested attack only.
    """
    try:
        return await get_message_send_service().get_status_async(
            attack_result_id=attack_result_id, send_id=send_id, wait_ms=wait_ms
        )
    except MessageSendNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
