# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Progressive topology and scoped, opt-in content routes for objective conversations."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from pyrit.backend.models.conversation_tree import (
    ConversationTreePage,
    ConversationTreePreviewRequest,
    ConversationTreePreviewResponse,
)
from pyrit.backend.services.conversation_tree_service import (
    ConversationTreeService,
    TreeCursorExpiredError,
    TreeCursorMismatchError,
    get_conversation_tree_service,
)
from pyrit.memory.conversation_tree import TreeReadLimitError, TreeSnapshotChangedError

router = APIRouter(prefix="/attacks", tags=["attacks"])


@router.get("/{attack_result_id}/conversation-tree", response_model=ConversationTreePage)
async def get_conversation_tree_async(
    *,
    attack_result_id: UUID,
    service: Annotated[ConversationTreeService, Depends(get_conversation_tree_service)],
    cursor: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=ConversationTreeService.MAX_PAGE_MESSAGES)] = 100,
    prioritize_conversation_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> ConversationTreePage:
    """
    Return one bounded page, not a drained snapshot of every stored transcript.

    Returns:
        ConversationTreePage: Nodes and known complete endpoints for the next tree increment.

    Raises:
        HTTPException: If a cursor expired, conflicts with the request, or exceeds a read budget.
    """
    try:
        return await service.get_page_async(
            attack_result_id=str(attack_result_id),
            cursor=cursor,
            limit=limit,
            prioritize_conversation_id=prioritize_conversation_id,
        )
    except TreeCursorExpiredError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except (TreeCursorMismatchError, TreeSnapshotChangedError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except TreeReadLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc


@router.post("/{attack_result_id}/conversation-tree/previews", response_model=ConversationTreePreviewResponse)
async def get_conversation_tree_previews_async(
    *,
    attack_result_id: UUID,
    request: ConversationTreePreviewRequest,
    service: Annotated[ConversationTreeService, Depends(get_conversation_tree_service)],
) -> ConversationTreePreviewResponse:
    """
    Retrieve bounded all-piece previews after checking current active membership.

    Returns:
        ConversationTreePreviewResponse: Deduplicated, ordered previews of explicitly requested messages.

    Raises:
        HTTPException: If the atomic piece budget is exceeded.
    """
    try:
        return await service.get_previews_async(attack_result_id=str(attack_result_id), request=request)
    except TreeReadLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc


@router.get("/{attack_result_id}/conversation-tree/pieces/{piece_id}/thumbnail")
async def get_conversation_tree_thumbnail_async(
    *,
    attack_result_id: UUID,
    piece_id: UUID,
    service: Annotated[ConversationTreeService, Depends(get_conversation_tree_service)],
) -> Response:
    """
    Serve a real, bounded local PNG thumbnail, never a redirected full original.

    Returns:
        Response: Thumbnail bytes after scope, storage-path and decode-budget checks.
    """
    data = await service.get_thumbnail_async(attack_result_id=str(attack_result_id), piece_id=piece_id)
    return Response(content=data, media_type="image/png", headers={"Cache-Control": "private, max-age=60"})
