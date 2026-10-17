# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Opaque API dispatch and canonical original readback, with public fixtures only."""

from __future__ import annotations

import asyncio
import hashlib
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from inspect_ai.log import read_eval_log

from pyrit.backend.middleware.auth import AuthenticatedUser
from pyrit.backend.models.evaluation_jobs import EvaluationCaseLaunch
from pyrit.backend.routes import evaluation_jobs
from pyrit.backend.services.evaluation_job_application import (
    EvaluationJobApplication,
    install_evaluation_job_application,
)
from pyrit.backend.services.runtime_lifecycle import RuntimeLifecycle
from pyrit.executor.benchmark.inspect_original_eval import (
    InspectOriginalEvalImporter,
    InspectOriginalScorePolicy,
    InspectSuccessDirection,
)
from pyrit.executor.jobs.inspect import (
    ApprovedOriginalInspectBinding,
    OriginalInspectArtifactWriter,
    PublicOriginalInspectJobRuntime,
)
from pyrit.executor.jobs.local import LocalEvaluationJobPort
from pyrit.executor.jobs.port import (
    EvaluationJobError,
    EvaluationJobErrorCode,
    EvaluationJobRuntimeRegistry,
    EvaluationRuntimeArtifacts,
    EvaluationRuntimeCancelled,
    EvaluationRuntimeError,
)
from pyrit.models import AttackOutcome, InputVariantRef, ScoreStatus
from pyrit.models.evaluation_job import EvaluationCleanupState, EvaluationJobState

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from starlette.middleware.base import RequestResponseEndpoint
    from starlette.requests import Request
    from starlette.responses import Response

    from pyrit.executor.jobs.port import EvaluationRuntimeContext
    from pyrit.memory import SQLiteMemory

ACTOR = "00000000-0000-4000-8000-000000000001"
OTHER = "00000000-0000-4000-8000-000000000002"
DENIED = "00000000-0000-4000-8000-000000000003"
pytestmark = [
    pytest.mark.filterwarnings(r"ignore:MemoryInterface\.:DeprecationWarning"),
    pytest.mark.usefixtures("patch_central_database"),
]


@dataclass(kw_only=True)
class _InstalledFixture:
    application: EvaluationJobApplication
    runtime: PublicOriginalInspectJobRuntime
    binding: ApprovedOriginalInspectBinding
    app: FastAPI
    actor: list[str | None]


@pytest.fixture
async def installed(*, sqlite_instance: SQLiteMemory, tmp_path: Path) -> AsyncIterator[_InstalledFixture]:
    runtime = await PublicOriginalInspectJobRuntime.create_async()
    binding = ApprovedOriginalInspectBinding(
        case=runtime.source.case,
        spec=runtime.source.spec,
        score_policy=InspectOriginalScorePolicy(
            task_name=runtime.source.case.task_name,
            task_version=runtime.source.case.task_version,
            primary_scorer="original_inert_scorer",
        ),
    )
    writer = OriginalInspectArtifactWriter(memory=sqlite_instance, binding=binding)
    port = LocalEvaluationJobPort(
        root=tmp_path / "jobs",
        registry=EvaluationJobRuntimeRegistry((runtime,)),
        writers={binding.registration.runtime: writer},
        allowed_actor_ids=frozenset({ACTOR, OTHER, DENIED}),
    )
    application = EvaluationJobApplication(
        port=port, writer=writer, case_ref="public-original-fixture", allowed_actor_ids=frozenset({ACTOR, OTHER})
    )
    app = FastAPI()
    app.state.evaluation_job_port = port
    app.state.evaluation_job_application = application
    app.include_router(evaluation_jobs.router, prefix="/api")
    actor: list[str | None] = [ACTOR]

    @app.middleware("http")
    async def trusted_fixture_identity_async(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if actor[0] is not None:
            request.state.user = AuthenticatedUser(oid=actor[0], name="Public fixture", email="", groups=[])
        return await call_next(request)

    await port.startup_async()
    try:
        yield _InstalledFixture(application=application, runtime=runtime, binding=binding, app=app, actor=actor)
    finally:
        await port.shutdown_async()


async def _launch_async(*, client: httpx.AsyncClient, request_id: UUID | None = None) -> UUID:
    response = await client.post(
        "/api/evaluation-jobs/cases/public-original-fixture", json={"request_id": str(request_id or uuid4())}
    )
    assert response.status_code == 202, response.text
    return UUID(response.json()["job_id"])


async def _complete_async(*, installed: _InstalledFixture, client: httpx.AsyncClient) -> UUID:
    job_id = await _launch_async(client=client)
    installed.application.port.start_consumer()
    terminal = await installed.application.port.wait_async(job_id=job_id, actor_id=ACTOR)
    assert terminal.state is EvaluationJobState.SUCCEEDED and terminal.canonical
    return job_id


async def test_opaque_api_original_lifecycle_exact_retention_and_read_only_grade_async(
    *, installed: _InstalledFixture, sqlite_instance: SQLiteMemory
) -> None:
    port = installed.application.port
    archives: list[bytes] = []
    execute = installed.runtime.execute_async

    async def observed_execute_async(*, context: EvaluationRuntimeContext) -> EvaluationRuntimeArtifacts:
        artifacts = await execute(context=context)
        archives.append(artifacts.payloads[0][1])
        return artifacts

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=installed.app), base_url="http://test") as client:
        request_id = uuid4()
        with patch.object(installed.runtime, "execute_async", side_effect=observed_execute_async) as executed:
            job_id = await _launch_async(client=client, request_id=request_id)
            replay = await client.post(
                "/api/evaluation-jobs/cases/public-original-fixture", json={"request_id": str(request_id)}
            )
            assert replay.json()["duplicate"] and replay.json()["job_id"] == str(job_id)
            assert (await client.get("/api/evaluation-jobs/catalog")).json() == []
            queued = await port.status_async(job_id=job_id, actor_id=ACTOR)
            assert queued.request.source == installed.binding.case.package
            assert queued.request.execution_profile_sha256 == installed.binding.spec.spec_sha256
            assert queued.request.controls == ()
            executed.assert_not_awaited()
            assert (
                await client.post("/api/evaluation-jobs", json=queued.request.model_dump(mode="json"))
            ).status_code == 403
            port.start_consumer()
            terminal = await port.wait_async(job_id=job_id, actor_id=ACTOR)
            assert terminal.state is EvaluationJobState.SUCCEEDED and terminal.canonical
            executed.assert_awaited_once()
            with (
                patch.object(InspectOriginalEvalImporter, "import_eval_bytes_async", new_callable=AsyncMock) as write,
                patch.object(sqlite_instance, "add_score_attack_result_pairs_to_memory") as write_pairs,
            ):
                result = await client.get(f"/api/evaluation-jobs/{job_id}/result")
                archived = await client.get(f"/api/evaluation-jobs/{job_id}/archive")
                assert result.status_code == archived.status_code == 200
                assert (await client.get(f"/api/evaluation-jobs/{job_id}/result")).json() == result.json()
                write.assert_not_awaited()
                write_pairs.assert_not_called()
            view = result.json()
            assert view["original_score"] == "1.0" and view["pyrit_score_status"] == ScoreStatus.COMPLETE
            assert view["pyrit_attack_outcome"] == AttackOutcome.UNDETERMINED
            assert view["canonical"]["source_complete"] and view["cleanup"] == "verified"
            assert view["final_score_event_id"] and view["final_score_event_sha256"]
            assert "original_inert_scorer" not in result.text
            assert archived.content == archives[0]
            assert hashlib.sha256(archived.content).hexdigest() == terminal.canonical.artifact_sha256
            assert archived.headers["etag"] == f'"{terminal.canonical.artifact_sha256}"'
            assert archived.headers["cache-control"] == "no-store"
            assert result.headers["cache-control"] == "no-store"
            log = await asyncio.to_thread(read_eval_log, io.BytesIO(archived.content), format="eval")
            assert log.samples and log.samples[0].store["lifecycle"] == ["setup", "solver", "score", "cleanup"]
            assert not log.samples[0].model_usage
            scores = await sqlite_instance.get_scores_async(
                score_ids=[str(identifier) for identifier in terminal.canonical.score_ids]
            )
            results = await sqlite_instance.get_attack_results_async(
                attack_result_ids=[str(identifier) for identifier in terminal.canonical.attack_result_ids]
            )
            assert len(scores) == len(results) == 1
            assert results[0].automated_score and results[0].automated_score.id == scores[0].id
            assert scores[0].score_metadata["inspect_final_score_event_id"] == view["final_score_event_id"]
            replay = await client.post(
                "/api/evaluation-jobs/cases/public-original-fixture", json={"request_id": str(request_id)}
            )
            assert replay.status_code == 202 and replay.json()["duplicate"]
            executed.assert_awaited_once()
            assert (await port.status_async(job_id=job_id, actor_id=ACTOR)).last_sequence == terminal.last_sequence


async def test_launch_retry_identity_is_actor_and_case_scoped_and_no_auth_is_not_admission_async(
    installed: _InstalledFixture,
) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=installed.app), base_url="http://test") as client:
        retry_key = uuid4()
        first = await _launch_async(client=client, request_id=retry_key)
        installed.actor[0] = OTHER
        second = await _launch_async(client=client, request_id=retry_key)
        assert second != first
        for suffix in ("", "/result", "/archive"):
            assert (await client.get(f"/api/evaluation-jobs/{first}{suffix}")).status_code == 403
        assert (await client.post(f"/api/evaluation-jobs/{first}/cancel")).status_code == 403
        response = await client.post("/api/evaluation-jobs/cases/other-alias", json={"request_id": str(retry_key)})
        assert response.status_code == 404 and str(first) not in response.text
        installed.actor[0] = DENIED
        for path in ("/api/evaluation-jobs/catalog", f"/api/evaluation-jobs/{first}/archive"):
            assert (await client.get(path)).status_code == 403
        assert (
            await client.post("/api/evaluation-jobs/cases/public-original-fixture", json={"request_id": str(retry_key)})
        ).status_code == 403
        installed.actor[0] = None
        response = await client.post(
            "/api/evaluation-jobs/cases/public-original-fixture",
            json={"request_id": str(retry_key)},
            headers={"X-Actor-ID": ACTOR, "X-PyRIT-Operator": ACTOR},
        )
        assert response.status_code == 401


async def test_same_actor_case_retry_cannot_silently_replay_a_changed_installed_profile_async(
    installed: _InstalledFixture,
) -> None:
    launch = EvaluationCaseLaunch(request_id=uuid4())
    application = installed.application
    original = await application.submit_async(case_ref=application.case_ref, launch=launch, actor_id=ACTOR)
    changed = ApprovedOriginalInspectBinding(
        case=installed.binding.case,
        spec=installed.binding.spec.model_copy(
            update={"model_route": installed.binding.spec.model_route.model_copy(update={"config_sha256": "f" * 64})}
        ),
        score_policy=installed.binding.score_policy,
    )
    runtime = MagicMock(spec=PublicOriginalInspectJobRuntime)
    runtime.registration = changed.registration
    application.port.registry = EvaluationJobRuntimeRegistry((runtime,))
    writer = OriginalInspectArtifactWriter(memory=application.writer.canonical_memory, binding=changed)
    application.port._writers = {writer.registration.runtime: writer}
    replacement = EvaluationJobApplication(
        port=application.port, writer=writer, case_ref=application.case_ref, allowed_actor_ids=frozenset({ACTOR})
    )
    with pytest.raises(EvaluationJobError, match="request_conflict"):
        await replacement.submit_async(case_ref=replacement.case_ref, launch=launch, actor_id=ACTOR)
    retained = await application.port.status_async(job_id=original.job_id, actor_id=ACTOR)
    assert retained.request.execution_profile_sha256 == installed.binding.spec.spec_sha256
    runtime.execute_async.assert_not_awaited()


@pytest.mark.parametrize(
    "field",
    [
        "source",
        "case_id",
        "run_id",
        "attempt_id",
        "writer",
        "canonical",
        "profile",
        "task_code",
        "model",
        "credentials",
    ],
)
async def test_opaque_launch_refuses_producer_execution_inputs_before_dispatch_async(
    *, installed: _InstalledFixture, field: str
) -> None:
    with patch.object(installed.runtime, "execute_async", new_callable=AsyncMock) as executed:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=installed.app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/evaluation-jobs/cases/public-original-fixture",
                json={"request_id": str(uuid4()), field: "unapproved-input"},
            )
            assert response.status_code == 400 and "unapproved-input" not in response.text
            response = await client.post(
                "/api/evaluation-jobs/cases/public-original-fixture",
                params={field: "unapproved-input"},
                json={"request_id": str(uuid4())},
            )
            assert response.status_code == 400
        executed.assert_not_awaited()


@pytest.mark.parametrize("cancel", [False, True])
async def test_execution_error_or_joined_cancellation_never_creates_an_original_grade_async(
    *, installed: _InstalledFixture, sqlite_instance: SQLiteMemory, cancel: bool
) -> None:
    entered = asyncio.Event()

    async def controlled_execute_async(*, context: EvaluationRuntimeContext) -> EvaluationRuntimeArtifacts:
        entered.set()
        if not cancel:
            raise EvaluationRuntimeError(
                code=EvaluationJobErrorCode.RUNTIME_FAILED, cleanup=EvaluationCleanupState.VERIFIED
            )
        try:
            await asyncio.Future()
        except asyncio.CancelledError as error:
            raise EvaluationRuntimeCancelled(EvaluationCleanupState.VERIFIED) from error
        raise AssertionError("Cancelled fixture resumed execution.")

    with patch.object(installed.runtime, "execute_async", side_effect=controlled_execute_async):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=installed.app), base_url="http://test"
        ) as client:
            job_id = await _launch_async(client=client)
            port = installed.application.port
            port.start_consumer()
            await asyncio.wait_for(entered.wait(), 5)
            if cancel:
                assert (await client.post(f"/api/evaluation-jobs/{job_id}/cancel")).status_code == 200
            terminal = await port.wait_async(job_id=job_id, actor_id=ACTOR)
            assert terminal.state is (EvaluationJobState.CANCELLED if cancel else EvaluationJobState.FAILED)
            assert terminal.cleanup is EvaluationCleanupState.VERIFIED and terminal.canonical is None
            assert (await client.get(f"/api/evaluation-jobs/{job_id}/result")).status_code == 409
            assert (await client.get(f"/api/evaluation-jobs/{job_id}/archive")).status_code == 409
            assert await sqlite_instance.get_attack_results_async() == []


@pytest.mark.parametrize(
    "field",
    [
        "job_id",
        "request_sha256",
        "manifest_sha256",
        "artifact_sha256",
        "projection_id",
        "score_ids",
        "attack_result_ids",
        "source_complete",
    ],
)
async def test_retained_readback_refuses_substituted_canonical_identity_async(
    *, installed: _InstalledFixture, field: str
) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=installed.app), base_url="http://test") as client:
        job_id = await _complete_async(installed=installed, client=client)
        terminal = await installed.application.port.status_async(job_id=job_id, actor_id=ACTOR)
        assert terminal.canonical and terminal.manifest
        value: UUID | tuple[UUID, ...] | str | bool
        if field == "job_id":
            value = uuid4()
        elif field in {"score_ids", "attack_result_ids"}:
            value = (uuid4(),)
        elif field == "source_complete":
            value = False
        else:
            value = "foreign_projection" if field == "projection_id" else "f" * 64
        changed = terminal.canonical.model_copy(update={field: value})
        with pytest.raises(EvaluationJobError, match="artifact_mismatch"):
            await installed.application.writer.readback_async(
                request=terminal.request, manifest=terminal.manifest, canonical=changed
            )


async def test_wrong_run_bytes_and_changed_canonical_result_refused_without_reimport_async(
    *, installed: _InstalledFixture, sqlite_instance: SQLiteMemory
) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=installed.app), base_url="http://test") as client:
        job_id = await _complete_async(installed=installed, client=client)
        terminal, retained = await installed.application.readback_async(job_id=job_id, actor_id=ACTOR)
        assert terminal.manifest and terminal.canonical
        with patch.object(InspectOriginalEvalImporter, "import_eval_bytes_async", new_callable=AsyncMock) as importer:
            with pytest.raises(EvaluationJobError, match="artifact_mismatch"):
                await installed.application.writer.readback_async(
                    request=terminal.request.model_copy(update={"run_id": uuid4()}),
                    manifest=terminal.manifest,
                    canonical=terminal.canonical,
                )
            with pytest.raises(EvaluationJobError, match="artifact_mismatch"):
                await installed.application.writer.import_async(
                    request=terminal.request,
                    artifacts=EvaluationRuntimeArtifacts(
                        manifest=terminal.manifest,
                        payloads=((terminal.manifest.artifacts[0].name, retained.archive[:-1]),),
                        cleanup=EvaluationCleanupState.VERIFIED,
                    ),
                )
            changed = await sqlite_instance.update_attack_result_by_id_async(
                attack_result_id=str(terminal.canonical.attack_result_ids[0]),
                update_fields={"outcome": AttackOutcome.SUCCESS},
            )
            assert changed
            response = await client.get(f"/api/evaluation-jobs/{job_id}/result")
            assert response.status_code == 409 and response.json()["detail"] == "artifact_mismatch"
            assert (await client.get(f"/api/evaluation-jobs/{job_id}/archive")).status_code == 409
            importer.assert_not_awaited()


def test_approved_original_metadata_and_installation_refuse_ambiguous_paths(installed: _InstalledFixture) -> None:
    binding = installed.binding
    with pytest.raises(ValueError, match="matching unchanged"):
        ApprovedOriginalInspectBinding(
            case=binding.case,
            spec=binding.spec.model_copy(
                update={
                    "input_variant": InputVariantRef(
                        case_id=binding.case.case_id, surface_id="input", content_sha256="a" * 64
                    )
                }
            ),
            score_policy=binding.score_policy,
        )
    with pytest.raises(ValueError, match="matching unchanged"):
        ApprovedOriginalInspectBinding(
            case=binding.case,
            spec=binding.spec,
            score_policy=InspectOriginalScorePolicy(
                task_name="wrong_task", task_version=binding.case.task_version, primary_scorer="unreviewed"
            ),
        )
    with pytest.raises(ValueError, match="matching unchanged"):
        ApprovedOriginalInspectBinding(
            case=binding.case,
            spec=binding.spec.model_copy(
                update={"package": binding.spec.package.model_copy(update={"source_sha256": "c" * 64})}
            ),
            score_policy=binding.score_policy,
        )
    with pytest.raises(ValueError, match="matching unchanged"):
        ApprovedOriginalInspectBinding(
            case=binding.case,
            spec=binding.spec,
            score_policy=InspectOriginalScorePolicy(
                task_name=binding.case.task_name,
                task_version=binding.case.task_version,
                primary_scorer=binding.score_policy.primary_scorer,
                success_direction=InspectSuccessDirection.AT_LEAST,
                success_threshold=0.5,
            ),
        )
    for runtime, approved in ((None, None), (installed.runtime, binding)):
        with pytest.raises(ValueError, match="exactly one"):
            OriginalInspectArtifactWriter(
                memory=installed.application.writer.canonical_memory, runtime=runtime, binding=approved
            )
    app = FastAPI()
    factory = AsyncMock()
    install_evaluation_job_application(app=app, factory=factory)
    with pytest.raises(ValueError, match="before lifespan"):
        install_evaluation_job_application(app=app, factory=factory)
    app.state.runtime_lifecycle = MagicMock(spec=RuntimeLifecycle)
    with pytest.raises(ValueError, match="before lifespan"):
        install_evaluation_job_application(app=app, factory=AsyncMock())
