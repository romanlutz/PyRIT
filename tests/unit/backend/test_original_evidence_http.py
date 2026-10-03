# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Worker capability authentication must never become browser/model Graph authentication."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.requests import Request
from starlette.responses import JSONResponse

from pyrit.backend.middleware.auth import EntraAuthMiddleware


async def test_exact_worker_post_does_not_forward_intake_capability_to_graph() -> None:
    with patch.dict(
        "os.environ",
        {
            "ENTRA_TENANT_ID": "public-fixture-tenant",
            "ENTRA_CLIENT_ID": "public-fixture-client",
            "ENTRA_ALLOWED_GROUP_IDS": "public-fixture-group",
        },
    ):
        middleware = EntraAuthMiddleware(MagicMock())
    route = f"/api/internal/original-evidence/{uuid.uuid4()}"
    request = Request(
        {
            "type": "http",
            "path": route,
            "method": "POST",
            "headers": [(b"authorization", b"Bearer public-fixture-intake-token")],
        }
    )
    authenticated = AsyncMock()
    next_handler = AsyncMock(return_value=JSONResponse({"detail": "worker-route-auth-required"}, status_code=401))
    with patch.object(middleware, "_authenticate_with_graph_async", authenticated):
        response = await middleware.dispatch(request=request, call_next=next_handler)
    assert response.status_code == 401
    authenticated.assert_not_called()
    next_handler.assert_called_once()


async def test_neighboring_api_cannot_use_the_worker_authentication_boundary() -> None:
    with patch.dict(
        "os.environ",
        {
            "ENTRA_TENANT_ID": "public-fixture-tenant",
            "ENTRA_CLIENT_ID": "public-fixture-client",
            "ENTRA_ALLOWED_GROUP_IDS": "public-fixture-group",
        },
    ):
        middleware = EntraAuthMiddleware(MagicMock())
    request = Request(
        {
            "type": "http",
            "path": "/api/internal/original-evidence/not-an-opaque-job",
            "method": "POST",
            "headers": [],
        }
    )
    next_handler = AsyncMock()
    response = await middleware.dispatch(request=request, call_next=next_handler)
    assert response.status_code == 401
    next_handler.assert_not_called()
