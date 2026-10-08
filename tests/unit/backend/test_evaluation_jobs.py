# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Default-off authenticated API/client and startup-owned local queue boundaries."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from pyrit.backend.middleware.auth import AuthenticatedUser
from pyrit.backend.routes import evaluation_jobs
from pyrit.backend.services.evaluation_job_service import LocalEvaluationJobSettings
from pyrit.executor.jobs.client import EvaluationJobHttpClient
from pyrit.executor.jobs.inspect import create_public_original_job_port_async
from pyrit.executor.jobs.port import EvaluationJobError
from pyrit.memory import MemoryInterface
from pyrit.models.evaluation_job import EvaluationJobRequest, EvaluationJobState, EvaluationRuntimeKind

if TYPE_CHECKING:
    from pathlib import Path

    from starlette.middleware.base import RequestResponseEndpoint
    from starlette.requests import Request
    from starlette.responses import Response

    from pyrit.memory import SQLiteMemory

ACTOR = "00000000-0000-4000-8000-000000000001"
FOREIGN = "00000000-0000-4000-8000-000000000002"
pytestmark = pytest.mark.filterwarnings(r"ignore:MemoryInterface\.:DeprecationWarning")


async def test_job_routes_are_disabled_by_default_and_disabled_auth_does_not_grant_actor_async() -> None:
    app = FastAPI()
    app.include_router(evaluation_jobs.router, prefix="/api")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        response = await client.get("/api/evaluation-jobs/catalog")
        assert response.status_code == 503


async def test_authenticated_api_client_queue_to_canonical_sqlite_and_foreign_actor_refusal_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    port = await create_public_original_job_port_async(
        root=tmp_path / "queue", memory=sqlite_instance, allowed_actor_ids=frozenset({ACTOR, FOREIGN})
    )
    await port.startup_async()
    app = FastAPI()
    app.state.evaluation_job_port = port
    app.include_router(evaluation_jobs.router, prefix="/api")
    active_actor: list[str | None] = [None]

    @app.middleware("http")
    async def authenticated_fixture_async(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if active_actor[0] is not None:
            request.state.user = AuthenticatedUser(oid=active_actor[0], name="Fixture", email="", groups=[])
        return await call_next(request)

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as http:
            assert (await http.get("/api/evaluation-jobs/catalog")).status_code == 401
            active_actor[0] = ACTOR
            client = EvaluationJobHttpClient(client=http, actor_id=ACTOR)
            registrations = await client.catalog_async()
            assert len(registrations) == 1 and registrations[0].runtime is EvaluationRuntimeKind.ORIGINAL_INSPECT
            assert registrations[0].controls == ()
            registration = registrations[0]
            request = EvaluationJobRequest(
                job_id=uuid4(),
                run_id=uuid4(),
                attempt_id=uuid4(),
                runtime=registration.runtime,
                source=registration.source,
                case_id=registration.case_id,
                execution_profile_sha256=registration.execution_profile_sha256,
            )
            submission = await client.submit_async(request=request, actor_id=ACTOR)
            assert submission.request_sha256 == request.request_sha256
            assert (await client.status_async(job_id=request.job_id, actor_id=ACTOR)).state is EvaluationJobState.QUEUED
            active_actor[0] = FOREIGN
            response = await http.get(f"/api/evaluation-jobs/{request.job_id}")
            assert response.status_code == 403 and response.json()["detail"] == "not_authorized"
            response = await http.post(f"/api/evaluation-jobs/{request.job_id}/cancel")
            assert response.status_code == 403
            active_actor[0] = ACTOR
            with pytest.raises(EvaluationJobError):
                await client.submit_async(request=request, actor_id=FOREIGN)
            for extra in ("input", "source_url", "task_code", "model", "credentials"):
                value = request.model_dump(mode="json")
                value[extra] = "not approved"
                response = await http.post("/api/evaluation-jobs", json=value)
                assert response.status_code == 400
                assert "not approved" not in response.text
            for runtime_kind in (EvaluationRuntimeKind.INSPECT_VARIANT, EvaluationRuntimeKind.NATIVE_BINDING):
                unsupported = request.model_copy(update={"runtime": runtime_kind})
                response = await http.post("/api/evaluation-jobs", json=unsupported.model_dump(mode="json"))
                assert response.status_code == 400 and response.json()["detail"] == "unsupported_runtime"
            response = await http.post(
                "/api/evaluation-jobs", content=b"x" * 32769, headers={"Content-Type": "application/json"}
            )
            assert response.status_code == 413
            port.start_consumer()
            terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
            status = await client.status_async(job_id=request.job_id, actor_id=ACTOR)
            assert status == terminal
            assert status.state is EvaluationJobState.SUCCEEDED and status.canonical
            assert status.canonical.source_complete
            assert len(status.canonical.score_ids) == len(status.canonical.attack_result_ids) == 1
            response = await http.get(f"/api/evaluation-jobs/{request.job_id}", params={"after_sequence": 999})
            assert response.status_code == 409
    finally:
        await port.shutdown_async()


@pytest.mark.parametrize(
    "environment",
    [
        {"PYRIT_EVALUATION_JOB_BACKEND": "azure"},
        {"PYRIT_EVALUATION_JOB_BACKEND": "local"},
        {"PYRIT_EVALUATION_JOB_ROOT": "relative"},
        {
            "PYRIT_EVALUATION_JOB_BACKEND": "local",
            "PYRIT_EVALUATION_JOB_ROOT": "relative",
            "PYRIT_EVALUATION_JOB_ALLOWED_OPERATOR_OIDS": ACTOR,
        },
    ],
)
def test_local_job_settings_reject_partial_or_unsupported_configuration(environment: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        LocalEvaluationJobSettings.from_environment(environment)


def test_local_job_settings_are_explicit_and_default_off(tmp_path: Path) -> None:
    assert LocalEvaluationJobSettings.from_environment({}) is None
    settings = LocalEvaluationJobSettings.from_environment(
        {
            "PYRIT_EVALUATION_JOB_BACKEND": "local",
            "PYRIT_EVALUATION_JOB_ROOT": str(tmp_path),
            "PYRIT_EVALUATION_JOB_ALLOWED_OPERATOR_OIDS": ACTOR,
        }
    )
    assert settings and settings.root == tmp_path and settings.allowed_actor_ids == frozenset({ACTOR})


async def test_local_job_settings_refuse_non_sqlite_canonical_memory_async(tmp_path: Path) -> None:
    settings = LocalEvaluationJobSettings(root=tmp_path, allowed_actor_ids=frozenset({ACTOR}))
    with pytest.raises(ValueError, match="SQLite"):
        await settings.create_port_async(memory=MagicMock(spec=MemoryInterface))


async def test_http_client_rejects_foreign_receipt_and_insecure_endpoint_async() -> None:
    async with httpx.AsyncClient(base_url="http://example.invalid") as client:
        with pytest.raises(ValueError):
            EvaluationJobHttpClient(client=client, actor_id=ACTOR)
    async with httpx.AsyncClient(base_url="https://example.invalid") as http:
        client = EvaluationJobHttpClient(client=http, actor_id=ACTOR)
        request = EvaluationJobRequest.model_validate_json(
            json.dumps(
                {
                    "job_id": str(uuid4()),
                    "run_id": str(uuid4()),
                    "attempt_id": str(uuid4()),
                    "runtime": "native_binding",
                    "source": {"kind": "named", "name": "public_fixture", "source_sha256": "a" * 64},
                    "case_id": "b" * 64,
                    "execution_profile_sha256": "c" * 64,
                }
            )
        )
        response = httpx.Response(
            202,
            request=httpx.Request("POST", "https://example.invalid"),
            json={"job_id": str(uuid4()), "request_sha256": request.request_sha256, "duplicate": False},
        )
        with patch.object(http, "post", new_callable=AsyncMock, return_value=response):
            with pytest.raises(EvaluationJobError, match="request_conflict"):
                await client.submit_async(request=request, actor_id=ACTOR)
