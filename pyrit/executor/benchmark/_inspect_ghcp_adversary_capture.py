# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Capture real PyRIT adversarial-target HTTP bytes without changing attack decisions."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class InspectGhcpAdversarialCapture:
    """A bounded caller-owned HTTP client for the PyRIT adversarial target."""

    def __init__(
        self,
        *,
        endpoint: str,
        max_body_bytes: int,
        sink: Callable[[str, str, bytes, int | None, str | None], Awaitable[None]],
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """
        Allow only the trusted local Chat Completions route, never external URLs.

        Raises:
            ValueError: If the adversarial model endpoint or capture cap is unsafe.
        """
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.port is None
            or parsed.path != "/v1/chat/completions"
            or parsed.username is not None
            or max_body_bytes < 1
        ):
            raise ValueError("PyRIT adversarial HTTP must use a bounded trusted loopback-only chat route.")
        self._endpoint = endpoint
        self._max_body_bytes = max_body_bytes
        self._sink = sink
        self._pending: dict[int, str] = {}
        self._closed = False
        self.request_count = 0
        self.response_count = 0
        self.client = httpx.AsyncClient(
            transport=transport,
            timeout=60,
            trust_env=False,
            follow_redirects=False,
            event_hooks={
                "request": [self._request_async],
                "response": [self._response_async],
            },
        )

    async def close_async(self) -> None:
        """Record missing responses explicitly and close the target's HTTP transport."""
        if self._closed:
            return
        self._closed = True
        await self.client.aclose()
        for request_id in self._pending.values():
            await self._sink(request_id, "error", b"", None, "adversarial_model_transport_ended")
        self._pending.clear()

    async def _request_async(self, request: httpx.Request) -> None:
        if request.method != "POST" or str(request.url) != self._endpoint:
            raise ValueError("PyRIT adversarial target attempted an unapproved local or remote HTTP route.")
        body = await request.aread()
        if not body or len(body) > self._max_body_bytes:
            raise ValueError("PyRIT adversarial request bytes are missing or exceed the approved quota.")
        request_id = str(uuid.uuid4())
        self._pending[id(request)] = request_id
        self.request_count += 1
        await self._sink(request_id, "request", body, None, None)

    async def _response_async(self, response: httpx.Response) -> None:
        body = await response.aread()
        request_id = self._pending.pop(id(response.request), None)
        if request_id is None:
            raise ValueError("PyRIT adversarial model response has no observed request.")
        if len(body) > self._max_body_bytes:
            await self._sink(request_id, "error", b"", response.status_code, "response_byte_quota_exceeded")
            raise ValueError("PyRIT adversarial model response bytes exceed the approved quota.")
        self.response_count += int(response.status_code == 200)
        await self._sink(
            request_id,
            "response",
            body,
            response.status_code,
            "host_adversarial_model_rejected" if response.status_code != 200 else None,
        )
