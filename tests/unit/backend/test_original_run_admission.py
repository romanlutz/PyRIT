# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""HTTP/SQLite admission tests with a genuine public original Task in a separate process."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from pyrit.backend.main import app
from pyrit.backend.middleware.auth import AuthenticatedUser
from pyrit.backend.services import original_run_admission as admission_module
from pyrit.backend.services.original_run_admission import (
    APPROVED_ORIGINAL_SCENARIO,
    OriginalAdmissionError,
    OriginalCleanupReceipt,
    OriginalRunBinding,
    OriginalRunGrant,
    OriginalRunVerification,
    OriginalWorkerJob,
    bind_original_worker_grant,
    get_bound_original_run_grant,
    install_trusted_original_runner,
)
from pyrit.backend.services.scenario_run_service import ScenarioRunService, _ActiveTask
from pyrit.backend.services.scenario_service import ScenarioService
from pyrit.models import AttackOutcome, ScenarioRunState, ScoreStatus, config_hash
from pyrit.models.catalog.scenario import OriginalRunReason, RunScenarioRequest
from pyrit.registry import ScenarioRegistry
from pyrit.scenario.scenarios.benchmark.inspect_original_inert import InspectOriginalInertScenario

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.types import Receive, Scope, Send

    from pyrit.memory import SQLiteMemory
    from pyrit.models import ScenarioResult

_WORKER_SCRIPT = Path(__file__).with_name("original_worker_process.py")
_WORKER_PROOF_PREFIX = "ORIGINAL_FIXTURE_PROOF:"
_OPERATOR = AuthenticatedUser(oid="reviewed-operator", name="Operator", email="operator@example.test", groups=[])
_OTHER_OPERATOR = AuthenticatedUser(oid="unapproved-operator", name="Other", email="other@example.test", groups=[])


@dataclass(frozen=True, kw_only=True)
class _FixtureGrant:
    """A host-only reservation with no private source identifier."""

    profile_ref: str
    operator_oid: str


@dataclass(kw_only=True)
class _FixtureJob:
    """Test broker-owned process and private typed proof; none enters backend SQLite."""

    job: OriginalWorkerJob
    binding: OriginalRunBinding
    process: asyncio.subprocess.Process | None = None
    source: dict[str, Any] | None = None
    cleanup: OriginalCleanupReceipt | None = None


class _FixtureRunner:
    """Launch only one fixed public fixture in another process and return a bounded proof."""

    profile_ref = "approved_public_fixture"
    model_role: Literal["evaluated"] = "evaluated"
    launch_mode: Literal["separate_process"] = "separate_process"

    def __init__(self) -> None:
        """Start closed and avoid publishing or qualifying any private runtime."""
        self.ready = False
        self.busy = False
        self.fail_readiness = False
        self.fail_before_source = False
        self.fail_after_source = False
        self.missing_terminal_digest = False
        self.readback_missing = False
        self.cleanup_proved = True
        self.display_score_override: str | None = None
        self.prepare_gate: asyncio.Event | None = None
        self.prepare_started: asyncio.Event | None = None
        self.worker_started: asyncio.Event | None = None
        self.worker_gate: asyncio.Event | None = None
        self.released_event: asyncio.Event | None = None
        self.acquired = 0
        self.released = 0
        self.worker_pid: int | None = None
        self._jobs: dict[uuid.UUID, _FixtureJob] = {}
        self._released: set[uuid.UUID] = set()

    def is_authorized(self, *, operator: AuthenticatedUser) -> bool:
        """Allow only the fixed test operator."""
        return operator.oid == _OPERATOR.oid

    def allows_display_score(self, *, value: str) -> bool:
        """Allow only the exact grade produced by the public pinned Task."""
        return value == "1.0"

    async def readiness_async(self, *, operator: AuthenticatedUser) -> tuple[OriginalRunReason, ...]:
        """Return only finite conditions, never a host URL or credential."""
        if self.fail_readiness:
            raise RuntimeError("http://internal.invalid/token=do-not-expose")
        if not self.is_authorized(operator=operator):
            return (OriginalRunReason.OPERATOR_NOT_AUTHORIZED,)
        return () if self.ready else (OriginalRunReason.PROVIDER_UNQUALIFIED,)

    async def acquire_async(self, *, operator: AuthenticatedUser) -> OriginalRunGrant:
        """Reserve one test-only slot; there is no request-supplied process or source."""
        if not self.ready:
            raise OriginalAdmissionError(reason=OriginalRunReason.PROVIDER_UNQUALIFIED)
        if self.busy:
            raise OriginalAdmissionError(reason=OriginalRunReason.CAPACITY_BUSY)
        self.busy = True
        self.acquired += 1
        return _FixtureGrant(profile_ref=self.profile_ref, operator_oid=operator.oid)

    async def prepare_worker_async(
        self, *, grant: OriginalRunGrant, job: OriginalWorkerJob, binding: OriginalRunBinding
    ) -> None:
        """Bind a new opaque job; only the fixed executable can run its public Task."""
        assert grant.operator_oid == binding.operator_oid and grant.profile_ref == binding.profile_ref
        self._jobs[job.job_ref] = _FixtureJob(job=job, binding=binding)
        if self.prepare_started is not None:
            self.prepare_started.set()
        if self.prepare_gate is not None:
            await self.prepare_gate.wait()

    async def start_worker_async(self, *, grant: OriginalRunGrant, job: OriginalWorkerJob) -> None:
        """Start the independent test worker, with its own pre-import roots and SQLite."""
        assert grant.operator_oid == self._jobs[job.job_ref].binding.operator_oid
        if not self.fail_before_source:
            root = _WORKER_SCRIPT.parents[3]
            inherited = {key: value for key, value in os.environ.items() if key in {"PATH", "SYSTEMROOT", "WINDIR"}}
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(_WORKER_SCRIPT),
                cwd=str(root),
                env=inherited,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._jobs[job.job_ref].process = process
            self.worker_pid = process.pid
        if self.worker_started is not None:
            self.worker_started.set()

    async def wait_worker_async(self, *, grant: OriginalRunGrant, job: OriginalWorkerJob) -> None:
        """Wait for process exit and accept only its typed, internally verified public proof."""
        _ = grant
        if self.fail_before_source:
            raise RuntimeError("http://internal.invalid/worker/secret")
        if self.worker_gate is not None:
            await self.worker_gate.wait()
        process = self._jobs[job.job_ref].process
        assert process is not None
        output, _ = await asyncio.wait_for(process.communicate(), timeout=60)
        if process.returncode != 0:
            raise RuntimeError("The isolated public test worker failed before a verifiable source result.")
        proofs = [
            line.removeprefix(_WORKER_PROOF_PREFIX)
            for line in output.decode("utf-8", errors="replace").splitlines()
            if line.startswith(_WORKER_PROOF_PREFIX)
        ]
        if len(proofs) != 1:
            raise RuntimeError("The isolated public test worker did not provide one source proof.")
        self._jobs[job.job_ref].source = json.loads(proofs[0])
        if self.fail_after_source:
            raise RuntimeError("The public worker returned a grade but its run did not finish successfully.")

    async def abort_worker_async(self, *, grant: OriginalRunGrant, job: OriginalWorkerJob) -> None:
        """Stop only this child PID before observing closure."""
        _ = grant
        process = self._jobs[job.job_ref].process
        if process is not None and process.returncode is None:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=10)

    async def release_async(
        self, *, grant: OriginalRunGrant, job: OriginalWorkerJob | None, cancelled: bool
    ) -> OriginalCleanupReceipt:
        """Observe closure exactly once; never create source evidence during cleanup."""
        _ = grant, cancelled
        if job is None:
            self.busy = False
            return OriginalCleanupReceipt(state="uncontained")
        if job.job_ref in self._released:
            return self._jobs[job.job_ref].cleanup or OriginalCleanupReceipt(state="uncontained")
        state = self._jobs.get(job.job_ref)
        process = state.process if state else None
        proved = self.cleanup_proved and (process is None or process.returncode is not None)
        receipt = (
            OriginalCleanupReceipt(state="proved", receipt_id=f"public-fixture-{job.job_ref.hex}")
            if proved
            else OriginalCleanupReceipt(state="uncontained")
        )
        if state is not None:
            state.cleanup = receipt
        self._released.add(job.job_ref)
        self.busy = False
        self.released += 1
        if self.released_event is not None:
            self.released_event.set()
        return receipt

    def verify_cleanup(self, *, job: OriginalWorkerJob) -> OriginalCleanupReceipt | None:
        """Read only the broker's independently retained exact closure receipt."""
        state = self._jobs.get(job.job_ref)
        return state.cleanup if state is not None else None

    def verify_result(self, *, job: OriginalWorkerJob) -> OriginalRunVerification | None:
        """Return a broker-bound proof; the web backend cannot open this worker's SQLite."""
        state = self._jobs.get(job.job_ref)
        if state is None or state.source is None or self.readback_missing:
            return None
        source = state.source
        digest = config_hash(
            {"job_ref": str(state.job.job_ref), "worker": source, "app_run_id": str(state.job.app_run_id)}
        )
        assert state.cleanup is not None
        return OriginalRunVerification(
            app_run_id=state.job.app_run_id,
            job_ref=state.job.job_ref,
            profile_ref=state.binding.profile_ref,
            operator_oid=state.binding.operator_oid,
            source_state="success",
            source_coverage_complete=source["source_coverage_complete"],
            original_score=self.display_score_override or source["original_score"],
            pyrit_score_status=ScoreStatus(source["pyrit_score_status"]),
            pyrit_outcome=AttackOutcome(source["pyrit_outcome"]),
            archive_sha256=source["archive_sha256"],
            final_score_event_id=source["final_score_event_id"],
            final_score_event_sha256=source["final_score_event_sha256"],
            operation_terminal_receipt_id=f"public-fixture-process-exit-{state.process.pid}",
            operation_terminal_sha256=(
                None
                if self.missing_terminal_digest
                else config_hash(
                    {
                        "app_run_id": str(state.job.app_run_id),
                        "job_ref": str(state.job.job_ref),
                        "archive_sha256": source["archive_sha256"],
                        "process_exit": state.process.returncode,
                    }
                )
            ),
            source_score_id=uuid.UUID(source["source_score_id"]),
            source_attack_result_id=uuid.UUID(source["source_attack_result_id"]),
            cleanup_receipt_id=state.cleanup.receipt_id,
            persistence_verified=True,
            proof_sha256=digest,
        )

    async def close_async(self) -> None:
        """Clean up only test subprocesses left alive by a failed assertion."""
        for state in self._jobs.values():
            process = state.process
            if process is not None and process.returncode is None:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=10)


class _AuthenticatedApp:
    """Attach a scoped test identity before the disabled local-auth middleware."""

    def __init__(self, *, operator: AuthenticatedUser) -> None:
        """Retain one test identity."""
        self.operator = operator

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Dispatch a real ASGI request with the test operator."""
        scope["state"] = {"user": self.operator}
        await app(scope, receive, send)


@pytest.fixture
async def approved_fixture_runner(
    patch_central_database: object, sqlite_instance: SQLiteMemory
) -> AsyncIterator[tuple[_FixtureRunner, ScenarioRunService]]:
    """Install only a fixed out-of-process public fixture; never register a private web Scenario."""
    _ = patch_central_database, sqlite_instance
    runner = _FixtureRunner()
    with patch.object(admission_module, "_gateway", None):
        install_trusted_original_runner(runner=runner)
        service = ScenarioRunService()
        catalog = ScenarioService()
        with (
            patch("pyrit.backend.routes.scenarios.get_scenario_run_service", return_value=service),
            patch("pyrit.backend.routes.scenarios.get_scenario_service", return_value=catalog),
        ):
            try:
                yield runner, service
            finally:
                await service.shutdown_async()
                await runner.close_async()


async def _launch_fixture_async(*, client: AsyncClient) -> str:
    """Obtain the real catalog admission and launch its fixed isolated worker job."""
    detail = await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")
    assert detail.status_code == 200
    token = detail.json()["original_run_admission"]["admission_ref"]
    assert isinstance(token, str)
    response = await client.post(
        "/api/scenarios/runs",
        json={"scenario_name": APPROVED_ORIGINAL_SCENARIO, "original_admission_ref": token},
    )
    assert response.status_code == 202, response.text
    run_id = response.json()["scenario_result_id"]
    assert isinstance(run_id, str)
    return run_id


async def _wait_for_job_async(*, service: ScenarioRunService, run_id: str) -> None:
    """Wait on one test-owned task before reading a terminal HTTP projection."""
    active = service._active_tasks[run_id]
    assert active.task is not None
    await asyncio.wait_for(active.task, timeout=75)


@pytest.mark.usefixtures("patch_central_database")
async def test_original_worker_absent_and_in_process_runner_both_fail_closed() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        missing = await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")
        rejected = await client.post(
            "/api/scenarios/runs",
            json={"scenario_name": APPROVED_ORIGINAL_SCENARIO, "original_admission_ref": "a" * 40},
        )
        public_inert = await client.get("/api/scenarios/catalog/benchmark.inspect_original_inert")
    assert missing.status_code == 404
    assert rejected.status_code == 503 and rejected.json()["detail"] == "runner_not_configured"
    assert public_inert.status_code == 200
    runner = _FixtureRunner()
    runner.launch_mode = "same_process"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="separate trusted host process"):
        admission_module.OriginalRunGateway(runner=runner)


def test_worker_grant_is_not_bound_to_the_web_process() -> None:
    grant = _FixtureGrant(profile_ref="approved_public_fixture", operator_oid=_OPERATOR.oid)
    with pytest.raises(OriginalAdmissionError, match="runner_not_configured"):
        get_bound_original_run_grant()
    with bind_original_worker_grant(grant=grant):
        assert get_bound_original_run_grant() is grant
        with pytest.raises(OriginalAdmissionError, match="profile_not_admitted"):
            with bind_original_worker_grant(grant=grant):
                pass
    with pytest.raises(OriginalAdmissionError, match="runner_not_configured"):
        get_bound_original_run_grant()


async def test_one_admitted_public_worker_executes_in_another_process_with_no_web_source_rows(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService], sqlite_instance: SQLiteMemory
) -> None:
    runner, service = approved_fixture_runner
    runner.ready = True
    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        registry = ScenarioRegistry.get_registry_singleton()
        with patch.object(
            registry,
            "get_all_registered_class_metadata",
            return_value=[registry.get_class_metadata(InspectOriginalInertScenario)],
        ):
            catalog = await client.get("/api/scenarios/catalog?include_estimates=false")
        assert catalog.status_code == 200
        names = {item["scenario_name"] for item in catalog.json()["items"]}
        assert APPROVED_ORIGINAL_SCENARIO in names
        assert "test.approved_private_task" not in names
        refused = await client.post("/api/scenarios/runs", json={"scenario_name": "test.approved_private_task"})
        assert refused.status_code == 400 and runner.acquired == 0
        detail = await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")
        token = detail.json()["original_run_admission"]["admission_ref"]
        assert detail.json()["original_run_admission"]["status"] == "ready"
        async with AsyncClient(
            transport=ASGITransport(app=_AuthenticatedApp(operator=_OTHER_OPERATOR), raise_app_exceptions=False),
            base_url="http://test",
        ) as other:
            wrong_actor = await other.post(
                "/api/scenarios/runs",
                json={"scenario_name": APPROVED_ORIGINAL_SCENARIO, "original_admission_ref": token},
            )
        assert wrong_actor.status_code == 403 and runner.acquired == 0
        for unsupported in (
            {"target_name": "arbitrary-target"},
            {"scenario_params": {"task_path": "C:\\unapproved\\source.py"}},
            {"source_url": "https://example.invalid/unapproved"},
            {"initializer_args": {"target": {"token": "never-forward"}}},
            {"model_role": "unapproved"},
            {"labels": {"source": "unapproved"}},
        ):
            rejected = await client.post(
                "/api/scenarios/runs",
                json={"scenario_name": APPROVED_ORIGINAL_SCENARIO, "original_admission_ref": token, **unsupported},
            )
            assert rejected.status_code == 400 and runner.acquired == 0
            assert "never-forward" not in rejected.text and "example.invalid" not in rejected.text
        started = await client.post(
            "/api/scenarios/runs",
            json={"scenario_name": APPROVED_ORIGINAL_SCENARIO, "original_admission_ref": token},
        )
        assert started.status_code == 202, started.text
        run_id = started.json()["scenario_result_id"]
        await _wait_for_job_async(service=service, run_id=run_id)
        assert runner.worker_pid is not None and runner.worker_pid != os.getpid()
        assert runner.acquired == runner.released == 1 and not runner.busy
        with pytest.raises(OriginalAdmissionError, match="runner_not_configured"):
            get_bound_original_run_grant()

        detail = await client.get(f"/api/scenarios/runs/{run_id}")
        progress = await client.get(f"/api/scenarios/runs/{run_id}/progress")
        history = await client.get("/api/scenarios/runs?scenario_names=benchmark.approved_original")
        assert detail.status_code == progress.status_code == history.status_code == 200
        result = detail.json()
        assert result["scenario_name"] == "ServerApprovedOriginalScenario"
        assert result["scenario_registry_name"] == APPROVED_ORIGINAL_SCENARIO
        assert result["status"] == "COMPLETED"
        assert (result["total_attacks"], result["completed_attacks"], result["successful_attacks"]) == (1, 1, 0)
        assert result["objective_achieved_rate"] is None
        assert result["original_source_result"]["original_score"] == "1.0"
        assert result["original_source_result"]["pyrit_score_status"] == "complete"
        assert result["original_source_result"]["pyrit_outcome"] == "undetermined"
        assert result["original_source_result"]["cleanup_state"] == "proved"
        assert progress.json()["summary"]["overall"]["success_percentage"] is None
        assert progress.json()["summary"]["overall"]["completed"] == 1
        assert progress.json()["results"] == []
        assert history.json()["items"][0]["original_source_result"] == result["original_source_result"]

        [header] = sqlite_instance.get_scenario_results(scenario_result_ids=[run_id])
        assert header.scenario_identifier.params["execution_owner"] == "approved_original"
        assert header.attack_results == {}
        assert len(sqlite_instance.get_scores()) == 0
        assert len(sqlite_instance.get_attack_results()) == 0
        assert "original_inspect_import" not in header.metadata
        link = header.metadata["approved_original_evidence_link"]
        assert set(link) == {"job_ref", "proof_sha256"}
        assert len(link["proof_sha256"]) == 64
        worker_source = next(iter(runner._jobs.values())).source
        assert worker_source is not None
        assert worker_source["source_score_id"] not in str(header.metadata)
        assert worker_source["source_attack_result_id"] not in str(header.metadata)
        for response in (detail, progress, history):
            assert worker_source["source_score_id"] not in response.text
            assert worker_source["source_attack_result_id"] not in response.text
            assert worker_source["archive_sha256"] not in response.text
            assert "original_inspect_inert" not in response.text
        assert (await client.get(f"/api/scenarios/runs/{run_id}/results")).status_code == 409
        direct = await client.get(f"/api/attacks/{worker_source['source_attack_result_id']}")
        assert direct.status_code == 404
        replay = await client.post(
            "/api/scenarios/runs",
            json={"scenario_name": APPROVED_ORIGINAL_SCENARIO, "original_admission_ref": token},
        )
        assert replay.status_code == 400 and replay.json()["detail"] == "admission_expired"
        ordinary = await client.post("/api/scenarios/runs", json={"scenario_name": "benchmark.inspect_original_inert"})
        assert ordinary.status_code == 202, ordinary.text
        await _wait_for_job_async(service=service, run_id=ordinary.json()["scenario_result_id"])
        ordinary_detail = await client.get(f"/api/scenarios/runs/{ordinary.json()['scenario_result_id']}")
        assert ordinary_detail.status_code == 200 and ordinary_detail.json()["status"] == "COMPLETED"

    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OTHER_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as other:
        assert (await other.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")).status_code == 404
        for endpoint in (
            f"/api/scenarios/runs/{run_id}",
            f"/api/scenarios/runs/{run_id}/progress",
            "/api/scenarios/runs",
        ):
            denied = await other.get(endpoint)
            assert denied.status_code == 403


async def test_unqualified_and_expired_admission_does_not_create_a_worker(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService],
) -> None:
    runner, _ = approved_fixture_runner
    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        pending = await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")
        assert pending.json()["original_run_admission"]["unmet_conditions"] == ["provider_unqualified"]
        assert pending.json()["original_run_admission"]["admission_ref"] is None
        runner.ready = True
        token = (await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")).json()[
            "original_run_admission"
        ]["admission_ref"]
        runner.ready = False
        refused = await client.post(
            "/api/scenarios/runs",
            json={"scenario_name": APPROVED_ORIGINAL_SCENARIO, "original_admission_ref": token},
        )
        assert refused.status_code == 409 and refused.json()["detail"] == "provider_unqualified"
        with patch.object(admission_module, "_ADMISSION_LIFETIME_SECONDS", -1):
            runner.ready = True
            expired = (await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")).json()[
                "original_run_admission"
            ]["admission_ref"]
            denied = await client.post(
                "/api/scenarios/runs",
                json={"scenario_name": APPROVED_ORIGINAL_SCENARIO, "original_admission_ref": expired},
            )
        assert denied.status_code == 400 and denied.json()["detail"] == "admission_expired"
        assert runner.acquired == runner.released == 0


async def test_provider_errors_do_not_expose_host_details_in_responses_or_web_logs(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService], caplog: pytest.LogCaptureFixture
) -> None:
    runner, _ = approved_fixture_runner
    runner.ready = True
    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        token = (await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")).json()[
            "original_run_admission"
        ]["admission_ref"]
        runner.fail_readiness = True
        pending = await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")
        rejected = await client.post(
            "/api/scenarios/runs",
            json={"scenario_name": APPROVED_ORIGINAL_SCENARIO, "original_admission_ref": token},
        )
        assert pending.json()["original_run_admission"]["unmet_conditions"] == ["provider_unqualified"]
        assert pending.json()["original_run_admission"]["admission_ref"] is None
        assert rejected.status_code == 409 and rejected.json()["detail"] == "provider_unqualified"
        assert runner.acquired == 0
        assert "internal.invalid" not in pending.text + rejected.text + caplog.text
        assert "do-not-expose" not in pending.text + rejected.text + caplog.text


async def test_invalid_projected_envelope_releases_the_reserved_worker(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService],
    sqlite_instance: SQLiteMemory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner, _ = approved_fixture_runner
    runner.ready = True
    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        token = (await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")).json()[
            "original_run_admission"
        ]["admission_ref"]
        with patch(
            "pyrit.backend.services.scenario_run_service.ScenarioIdentifier",
            side_effect=ValueError("http://private.invalid/secret"),
        ):
            response = await client.post(
                "/api/scenarios/runs",
                json={"scenario_name": APPROVED_ORIGINAL_SCENARIO, "original_admission_ref": token},
            )
    assert response.status_code == 409 and response.json()["detail"] == "provider_unqualified"
    assert runner.acquired == runner.released == 1 and not runner.busy
    assert runner._jobs == {}
    assert sqlite_instance.get_scenario_results() == []
    assert "private.invalid" not in response.text + caplog.text


async def test_worker_failure_before_sample_keeps_physical_cleanup_distinct_from_grade(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService],
    sqlite_instance: SQLiteMemory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner, service = approved_fixture_runner
    runner.ready = True
    runner.fail_before_source = True
    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        run_id = await _launch_fixture_async(client=client)
        await _wait_for_job_async(service=service, run_id=run_id)
        detail = await client.get(f"/api/scenarios/runs/{run_id}")
        progress = await client.get(f"/api/scenarios/runs/{run_id}/progress")
        history = await client.get("/api/scenarios/runs")
        assert detail.status_code == progress.status_code == history.status_code == 200
        for item in (detail.json(), history.json()["items"][0]):
            assert item["status"] == "FAILED"
            assert item["completed_attacks"] == 0
            assert item["original_source_result"]["status"] == "failed_ungraded"
            assert item["original_source_result"]["original_score"] is None
            assert item["original_source_result"]["source_coverage_complete"] is False
            assert item["original_source_result"]["cleanup_state"] == "proved"
            assert item["error"] == "source_unverified"
        assert progress.json()["summary"]["overall"]["completed"] == 0
        assert progress.json()["run"]["original_source_result"]["cleanup_state"] == "proved"
        assert runner.worker_pid is None and runner.released == 1
        assert sqlite_instance.get_scores() == [] and sqlite_instance.get_attack_results() == []
        assert "internal.invalid" not in detail.text + progress.text + history.text
        assert "internal.invalid" not in caplog.text
        await asyncio.to_thread(
            sqlite_instance.update_scenario_metadata_fields,
            scenario_result_id=run_id,
            fields={OriginalCleanupReceipt.METADATA_KEY: OriginalCleanupReceipt(state="uncontained").model_dump()},
        )
        for endpoint in (
            f"/api/scenarios/runs/{run_id}",
            f"/api/scenarios/runs/{run_id}/progress",
            "/api/scenarios/runs",
        ):
            assert (await client.get(endpoint)).status_code == 409


async def test_missing_physical_cleanup_proof_cannot_claim_a_completed_grade(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService],
) -> None:
    runner, service = approved_fixture_runner
    runner.ready = True
    runner.cleanup_proved = False
    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        run_id = await _launch_fixture_async(client=client)
        await _wait_for_job_async(service=service, run_id=run_id)
        detail = await client.get(f"/api/scenarios/runs/{run_id}")
        assert detail.status_code == 200
        assert detail.json()["status"] == "FAILED"
        assert detail.json()["completed_attacks"] == 0
        assert detail.json()["original_source_result"]["original_score"] == "1.0"
        assert detail.json()["original_source_result"]["status"] == "cleanup_uncertain"
        assert detail.json()["original_source_result"]["cleanup_state"] == "uncontained"


async def test_source_grade_with_failed_execution_is_not_counted_as_a_completed_case(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService],
) -> None:
    runner, service = approved_fixture_runner
    runner.ready = True
    runner.fail_after_source = True
    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        run_id = await _launch_fixture_async(client=client)
        await _wait_for_job_async(service=service, run_id=run_id)
        for response in (
            await client.get(f"/api/scenarios/runs/{run_id}"),
            await client.get("/api/scenarios/runs?scenario_names=benchmark.approved_original"),
        ):
            assert response.status_code == 200
            result = response.json()["items"][0] if "items" in response.json() else response.json()
            assert result["status"] == "FAILED"
            assert result["completed_attacks"] == 0
            assert result["original_source_result"]["status"] == "failed_source_verified"
            assert result["original_source_result"]["original_score"] == "1.0"
            assert result["original_source_result"]["cleanup_state"] == "proved"
        progress = await client.get(f"/api/scenarios/runs/{run_id}/progress")
        assert progress.status_code == 200 and progress.json()["summary"]["overall"]["completed"] == 0


async def test_missing_terminal_operation_proof_shows_only_verified_cleanup(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService],
) -> None:
    runner, service = approved_fixture_runner
    runner.ready = True
    runner.missing_terminal_digest = True
    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        run_id = await _launch_fixture_async(client=client)
        await _wait_for_job_async(service=service, run_id=run_id)
        for response in (
            await client.get(f"/api/scenarios/runs/{run_id}"),
            await client.get("/api/scenarios/runs?scenario_names=benchmark.approved_original"),
        ):
            assert response.status_code == 200
            result = response.json()["items"][0] if "items" in response.json() else response.json()
            assert result["status"] == "FAILED"
            assert result["completed_attacks"] == 0
            assert result["original_source_result"]["status"] == "failed_ungraded"
            assert result["original_source_result"]["original_score"] is None
            assert result["original_source_result"]["cleanup_state"] == "proved"
        progress = await client.get(f"/api/scenarios/runs/{run_id}/progress")
        assert progress.status_code == 200 and progress.json()["summary"]["overall"]["completed"] == 0


async def test_cancelled_running_worker_aborts_exact_pid_before_releasing(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService], sqlite_instance: SQLiteMemory
) -> None:
    runner, service = approved_fixture_runner
    runner.ready = True
    runner.worker_gate = asyncio.Event()
    runner.worker_started = asyncio.Event()
    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        run_id = await _launch_fixture_async(client=client)
        await asyncio.wait_for(runner.worker_started.wait(), timeout=10)
        assert runner.worker_pid is not None and runner.worker_pid != os.getpid()
        cancelled = await client.post(f"/api/scenarios/runs/{run_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "CANCELLED"
        assert cancelled.json()["completed_attacks"] == 0
        assert cancelled.json()["original_source_result"]["original_score"] is None
        assert cancelled.json()["original_source_result"]["cleanup_state"] == "proved"
        assert runner.released == 1 and not runner.busy
        assert sqlite_instance.get_scores() == []


async def test_interrupted_web_process_cannot_infer_worker_cleanup_or_source_completion(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService],
) -> None:
    runner, _ = approved_fixture_runner
    runner.ready = True
    runner.worker_gate = asyncio.Event()
    runner.worker_started = asyncio.Event()
    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        run_id = await _launch_fixture_async(client=client)
        await asyncio.wait_for(runner.worker_started.wait(), timeout=10)
        restarted = ScenarioRunService()
        try:
            assert await restarted.reconcile_interrupted_runs_async() == 1
            with patch("pyrit.backend.routes.scenarios.get_scenario_run_service", return_value=restarted):
                detail = await client.get(f"/api/scenarios/runs/{run_id}")
                progress = await client.get(f"/api/scenarios/runs/{run_id}/progress")
                history = await client.get("/api/scenarios/runs?scenario_names=benchmark.approved_original")
            assert detail.status_code == progress.status_code == history.status_code == 200
            for result in (detail.json(), history.json()["items"][0]):
                assert result["status"] == "FAILED"
                assert result["completed_attacks"] == 0
                assert result["original_source_result"]["status"] == "cleanup_uncertain"
                assert result["original_source_result"]["original_score"] is None
                assert result["original_source_result"]["cleanup_state"] == "uncontained"
                assert result["error"] == "cleanup_pending"
            assert progress.json()["summary"]["overall"]["completed"] == 0
        finally:
            await restarted.shutdown_async()


async def test_cancellation_during_worker_preparation_waits_for_owned_release(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService], sqlite_instance: SQLiteMemory
) -> None:
    runner, service = approved_fixture_runner
    runner.ready = True
    runner.prepare_gate = asyncio.Event()
    runner.prepare_started = asyncio.Event()
    runner.released_event = asyncio.Event()
    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        token = (await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")).json()[
            "original_run_admission"
        ]["admission_ref"]
    launch = asyncio.create_task(
        service.start_run_async(
            request=RunScenarioRequest(scenario_name=APPROVED_ORIGINAL_SCENARIO, original_admission_ref=token),
            operator=_OPERATOR,
        )
    )
    await asyncio.wait_for(runner.prepare_started.wait(), timeout=10)
    launch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await launch
    assert runner.released == 0
    runner.prepare_gate.set()
    await asyncio.wait_for(runner.released_event.wait(), timeout=10)
    if service._original_cleanup_tasks:
        await asyncio.wait_for(asyncio.gather(*service._original_cleanup_tasks), timeout=10)
    assert runner.released == 1 and not runner.busy
    job_id = str(next(iter(runner._jobs.values())).job.app_run_id)
    [stored] = sqlite_instance.get_scenario_results(scenario_result_ids=[job_id])
    assert stored.scenario_run_state is ScenarioRunState.CANCELLED
    summary = service.get_run(scenario_result_id=job_id, operator=_OPERATOR)
    assert summary is not None
    assert summary.original_source_result is not None
    assert summary.original_source_result.cleanup_state == "proved"


async def test_cancellation_during_scheduler_enqueue_releases_only_the_owned_worker(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService], sqlite_instance: SQLiteMemory
) -> None:
    runner, service = approved_fixture_runner
    runner.ready = True
    runner.released_event = asyncio.Event()
    enqueue_started = asyncio.Event()
    enqueue_gate = asyncio.Event()
    original_enqueue = service._enqueue_run_async

    async def slow_enqueue_async(*, scheduled: _ActiveTask) -> None:
        enqueue_started.set()
        await enqueue_gate.wait()
        await original_enqueue(scheduled=scheduled)

    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        token = (await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")).json()[
            "original_run_admission"
        ]["admission_ref"]
    with patch.object(service, "_enqueue_run_async", new=slow_enqueue_async):
        launch = asyncio.create_task(
            service.start_run_async(
                request=RunScenarioRequest(scenario_name=APPROVED_ORIGINAL_SCENARIO, original_admission_ref=token),
                operator=_OPERATOR,
            )
        )
        await asyncio.wait_for(enqueue_started.wait(), timeout=10)
        launch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await launch
        enqueue_gate.set()
        await asyncio.wait_for(runner.released_event.wait(), timeout=10)
        if service._original_cleanup_tasks:
            await asyncio.wait_for(asyncio.gather(*service._original_cleanup_tasks), timeout=10)
    job_id = str(next(iter(runner._jobs.values())).job.app_run_id)
    [stored] = sqlite_instance.get_scenario_results(scenario_result_ids=[job_id])
    assert stored.scenario_run_state is ScenarioRunState.CANCELLED
    process = next(iter(runner._jobs.values())).process
    assert process is None or process.returncode is not None
    assert runner.acquired == runner.released == 1


async def test_queued_original_cancellation_does_not_start_the_worker(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService],
) -> None:
    runner, service = approved_fixture_runner
    runner.ready = True
    ordinary_started = asyncio.Event()
    ordinary_gate = asyncio.Event()

    async def blocked_public_run_async(self: InspectOriginalInertScenario) -> ScenarioResult:
        _ = self
        ordinary_started.set()
        await ordinary_gate.wait()
        raise AssertionError("The held public run should have been cancelled before executing.")

    try:
        with patch.object(InspectOriginalInertScenario, "run_async", new=blocked_public_run_async):
            async with AsyncClient(
                transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
                base_url="http://test",
            ) as client:
                public = await client.post(
                    "/api/scenarios/runs", json={"scenario_name": "benchmark.inspect_original_inert"}
                )
                assert public.status_code == 202
                await asyncio.wait_for(ordinary_started.wait(), timeout=10)
                queued_id = await _launch_fixture_async(client=client)
                queued = await client.get(f"/api/scenarios/runs/{queued_id}")
                assert queued.status_code == 200 and queued.json()["status"] == "QUEUED"
                cancelled = await client.post(f"/api/scenarios/runs/{queued_id}/cancel")
                assert cancelled.status_code == 200 and cancelled.json()["status"] == "CANCELLED"
                assert cancelled.json()["original_source_result"]["cleanup_state"] == "proved"
                assert runner.worker_pid is None and runner.acquired == runner.released == 1
                held = await client.post(f"/api/scenarios/runs/{public.json()['scenario_result_id']}/cancel")
                assert held.status_code == 200 and held.json()["status"] == "CANCELLED"
    finally:
        ordinary_gate.set()


async def test_unapproved_raw_grade_or_swapped_broker_link_fails_closed(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService], sqlite_instance: SQLiteMemory
) -> None:
    runner, service = approved_fixture_runner
    runner.ready = True
    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        run_id = await _launch_fixture_async(client=client)
        await _wait_for_job_async(service=service, run_id=run_id)
        header = sqlite_instance.get_scenario_result_header(scenario_result_id=run_id)
        assert header is not None
        original_link = header.metadata["approved_original_evidence_link"]
        assert (await client.get(f"/api/scenarios/runs/{run_id}")).status_code == 200
        await asyncio.to_thread(
            sqlite_instance.update_scenario_metadata_fields,
            scenario_result_id=run_id,
            fields={
                "approved_original_evidence_link": {
                    **original_link,
                    "proof_sha256": hashlib.sha256(b"other-job").hexdigest(),
                }
            },
        )
        for endpoint in (
            f"/api/scenarios/runs/{run_id}",
            f"/api/scenarios/runs/{run_id}/progress",
            "/api/scenarios/runs?scenario_names=benchmark.approved_original",
        ):
            response = await client.get(endpoint)
            assert response.status_code == 409
        await asyncio.to_thread(
            sqlite_instance.update_scenario_metadata_fields,
            scenario_result_id=run_id,
            fields={"approved_original_evidence_link": original_link},
        )
        runner.display_score_override = "AlphanumericSecretDoNotReturn"
        detail = await client.get(f"/api/scenarios/runs/{run_id}")
        assert detail.status_code == 409
        assert "AlphanumericSecretDoNotReturn" not in detail.text
        runner.display_score_override = None
        original = dict(header.metadata)
        stripped = {key: value for key, value in original.items() if key != OriginalRunBinding.METADATA_KEY}
        await asyncio.to_thread(sqlite_instance.update_scenario_metadata, scenario_result_id=run_id, metadata=stripped)
        for endpoint in (
            f"/api/scenarios/runs/{run_id}",
            f"/api/scenarios/runs/{run_id}/progress",
            "/api/scenarios/runs",
            f"/api/scenarios/runs/{run_id}/results",
        ):
            assert (await client.get(endpoint)).status_code == 409
        await asyncio.to_thread(sqlite_instance.update_scenario_metadata, scenario_result_id=run_id, metadata=original)
        assert (await client.get(f"/api/scenarios/runs/{run_id}")).status_code == 200


async def test_other_valid_worker_job_cannot_replace_an_original_run_proof(
    approved_fixture_runner: tuple[_FixtureRunner, ScenarioRunService], sqlite_instance: SQLiteMemory
) -> None:
    runner, service = approved_fixture_runner
    runner.ready = True
    async with AsyncClient(
        transport=ASGITransport(app=_AuthenticatedApp(operator=_OPERATOR), raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        first = await _launch_fixture_async(client=client)
        await _wait_for_job_async(service=service, run_id=first)
        second = await _launch_fixture_async(client=client)
        await _wait_for_job_async(service=service, run_id=second)
        first_header = sqlite_instance.get_scenario_result_header(scenario_result_id=first)
        second_header = sqlite_instance.get_scenario_result_header(scenario_result_id=second)
        assert first_header is not None and second_header is not None
        assert (await client.get(f"/api/scenarios/runs/{first}")).status_code == 200
        other_job = dict(second_header.metadata[OriginalWorkerJob.METADATA_KEY])
        other_job["app_run_id"] = first
        await asyncio.to_thread(
            sqlite_instance.update_scenario_metadata_fields,
            scenario_result_id=first,
            fields={
                OriginalWorkerJob.METADATA_KEY: other_job,
                "approved_original_evidence_link": second_header.metadata["approved_original_evidence_link"],
                OriginalCleanupReceipt.METADATA_KEY: second_header.metadata[OriginalCleanupReceipt.METADATA_KEY],
                "approved_original_source_result": second_header.metadata["approved_original_source_result"],
            },
        )
        for endpoint in (
            f"/api/scenarios/runs/{first}",
            f"/api/scenarios/runs/{first}/progress",
            "/api/scenarios/runs?scenario_names=benchmark.approved_original",
        ):
            response = await client.get(endpoint)
            assert response.status_code == 409, f"Worker B was accepted as A by {endpoint}"
            assert runner._jobs[uuid.UUID(other_job["job_ref"])].source is not None
        await asyncio.to_thread(
            sqlite_instance.update_scenario_metadata,
            scenario_result_id=first,
            metadata=first_header.metadata,
        )
        assert (await client.get(f"/api/scenarios/runs/{first}")).status_code == 200
