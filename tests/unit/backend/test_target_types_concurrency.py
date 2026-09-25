# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Concurrency regressions for target type routes."""

import asyncio
from threading import Event
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient

from pyrit.backend.main import app
from pyrit.backend.services.target_service import TargetService


async def test_health_remains_schedulable_during_cold_target_types() -> None:
    loop = asyncio.get_running_loop()
    discovery_started = asyncio.Event()
    discovery_release = Event()
    discovery_finished = Event()
    service = TargetService()

    def _blocking_metadata_discovery() -> list[object]:
        if not discovery_release.is_set():
            loop.call_soon_threadsafe(discovery_started.set)
            discovery_release.wait()
        discovery_finished.set()
        return []

    transport = ASGITransport(app=app)
    with (
        patch.object(service._registry, "get_all_registered_class_metadata", side_effect=_blocking_metadata_discovery),
        patch("pyrit.backend.routes.targets.get_target_service", return_value=service),
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            types_request = asyncio.create_task(client.get("/api/targets/types"))
            started_task = asyncio.create_task(discovery_started.wait())
            try:
                done, _ = await asyncio.wait(
                    {started_task, types_request},
                    timeout=10,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if types_request in done:
                    response = types_request.result()
                    raise AssertionError(
                        f"/api/targets/types returned HTTP {response.status_code} before metadata discovery started"
                    )
                assert started_task in done, "Target metadata discovery did not start within 10 seconds"
                health_response = await asyncio.wait_for(client.get("/api/health"), timeout=2)
                assert health_response.status_code == 200
                assert not discovery_finished.is_set()
            finally:
                discovery_release.set()
                started_task.cancel()
                await asyncio.gather(started_task, return_exceptions=True)
                types_response = await asyncio.wait_for(types_request, timeout=10)

    assert types_response.status_code == 200
