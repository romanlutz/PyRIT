# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import base64
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI

from pyrit.analytics import AttackResultAnalytics
from pyrit.backend.middleware.error_handlers import register_error_handlers
from pyrit.backend.routes.analytics import router
from pyrit.backend.services.analytics_service import get_analytics_service
from pyrit.exceptions.analytics_exception import (
    AnalyticsBusyException,
    AnalyticsDataException,
    AnalyticsTimeoutException,
)
from pyrit.models import AttackAnalyticsQuery, AttackOutcome
from unit.memory.test_attack_analytics import make_result


@pytest.fixture
def seeded_memory(sqlite_instance):
    sqlite_instance.add_attack_results_to_memory(
        attack_results=[make_result(), make_result(index=2, outcome=AttackOutcome.ERROR)]
    )
    return sqlite_instance


@pytest.fixture
async def client_async(seeded_memory):
    service = AttackResultAnalytics(memory=seeded_memory)
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_analytics_service] = lambda: service
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield client, service
    finally:
        await asyncio.to_thread(service.shutdown)


async def test_query_returns_sdk_statistics_without_scores_async(client_async):
    client, service = client_async
    response = await client.post("/api/analytics/attacks/query", json={})
    assert response.status_code == 200
    data = response.json()
    sdk = await service.query_async(query=AttackAnalyticsQuery())
    assert data["summary"] == sdk.model_dump(mode="json")["summary"]
    assert data["summary"]["total_results"] == 2
    assert data["summary"]["success_rate"] == 1.0
    assert data["summary"]["errors"] == 1
    assert len(data["results"]["items"]) == 2
    assert "score" not in str(data["results"]).lower()


async def test_outcome_filter_returns_asterisk_annotation_async(client_async):
    client, _ = client_async
    response = await client.post("/api/analytics/attacks/query", json={"filters": {"outcomes": ["error"]}})
    assert response.status_code == 200
    assert response.json()["outcome_filter_applied"]
    assert response.json()["summary"]["success_rate"] is None


async def test_results_route_does_not_invoke_report_async(client_async):
    client, service = client_async
    with patch.object(service, "query_async", side_effect=AssertionError("Report must not run")):
        first = await client.post("/api/analytics/attacks/results", json={"limit": 1})
        second = await client.post(
            "/api/analytics/attacks/results", json={"limit": 1, "cursor": first.json()["next_cursor"]}
        )
    assert first.status_code == second.status_code == 200
    assert first.json()["items"][0]["attack_result_id"] != second.json()["items"][0]["attack_result_id"]
    assert not second.json()["has_more"]


async def test_facet_route_returns_only_requested_options_async(client_async):
    client, _ = client_async
    response = await client.post("/api/analytics/attacks/facets", json={"dimension": {"name": "operation"}})
    assert response.status_code == 200
    assert [item["key"]["value"] for item in response.json()["items"]] == ["operation-a"]


@pytest.mark.parametrize("body", [{"axis_limit": 21}, {"group_by": {"name": "arbitrary"}}, {"scores": True}])
async def test_invalid_report_requests_are_rejected_async(client_async, body):
    client, _ = client_async
    response = await client.post("/api/analytics/attacks/query", json=body)
    assert response.status_code == 422
    assert response.json()["status"] == 422


@pytest.mark.parametrize(
    "error, status",
    [
        (AnalyticsBusyException(), 503),
        (AnalyticsTimeoutException(), 504),
        (AnalyticsDataException("Invalid metadata"), 500),
    ],
)
async def test_capacity_and_data_failures_are_explicit_async(client_async, error, status):
    client, service = client_async
    with patch.object(service, "query_async", new_callable=AsyncMock) as query:
        query.side_effect = error
        response = await client.post("/api/analytics/attacks/query", json={})
    assert response.status_code == status
    assert response.json()["status"] == status
    assert response.json()["detail"] == error.message
    assert "summary" not in response.json()
    if status == 503:
        assert response.headers["Retry-After"] == "1"


async def test_invalid_cursor_has_explicit_failure_async(client_async):
    client, _ = client_async
    response = await client.post("/api/analytics/attacks/results", json={"cursor": "not-a-cursor"})
    assert response.status_code == 400
    assert "cursor" in response.json()["detail"]


@pytest.mark.parametrize("identifier", [123, [], {}])
async def test_structurally_invalid_cursor_is_client_error_async(
    *, client_async: tuple[httpx.AsyncClient, AttackResultAnalytics], identifier: object
) -> None:
    client, _ = client_async
    first = await client.post("/api/analytics/attacks/results", json={"limit": 1})
    assert first.status_code == 200
    valid_cursor = first.json()["next_cursor"]
    payload = json.loads(base64.urlsafe_b64decode(valid_cursor + "=" * (-len(valid_cursor) % 4)))
    payload["i"] = identifier
    cursor = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    response = await client.post("/api/analytics/attacks/results", json={"cursor": cursor})
    assert response.status_code == 400
    assert "cursor" in response.json()["detail"]
