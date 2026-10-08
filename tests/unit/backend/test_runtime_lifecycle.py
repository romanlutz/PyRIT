# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Admission and fail-fast runtime replacement invariants."""

import asyncio
import os
from collections.abc import Generator
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import Request

import pyrit.backend.services.runtime_lifecycle as lifecycle_module
from pyrit.backend.middleware.auth import require_admin
from pyrit.backend.middleware.runtime import RuntimeAdmissionMiddleware
from pyrit.backend.routes import configuration, health
from pyrit.backend.services.configuration_file_service import ConfigurationFileService
from pyrit.backend.services.evaluation_job_service import LocalEvaluationJobSettings
from pyrit.backend.services.manual_send_scheduler import get_manual_send_scheduler
from pyrit.backend.services.original_worker_preflight import CohostPreflight
from pyrit.backend.services.runtime_lifecycle import RuntimeLifecycle
from pyrit.backend.services.scenario_run_service import ScenarioRunService
from pyrit.executor.jobs.local import LocalEvaluationJobPort
from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.setup.configuration_loader import ConfigurationLoader


@pytest.fixture
def runtime(tmp_path: Path) -> Generator[RuntimeLifecycle, None, None]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "memory_db_type: in_memory\nenable_live_reinitialization: true\n",
        encoding="utf-8",
    )
    source = ConfigurationFileService(config_file_value=str(config_path))
    app = FastAPI()
    with patch.dict(os.environ, {}, clear=True):
        service = RuntimeLifecycle(app=app, source=source)
    assert service.topology_supported
    service.state = "ready"
    service.generation = "original"
    app.state.runtime_lifecycle = service
    app.state.configuration_file_service = source
    app.add_middleware(RuntimeAdmissionMiddleware)
    app.include_router(configuration.router, prefix="/api")
    app.include_router(health.router, prefix="/api")
    app.dependency_overrides[require_admin] = lambda: None
    config = ConfigurationLoader(
        memory_db_type="in_memory",
        env_files=[],
        enable_live_reinitialization=True,
    )
    prepared = MagicMock()
    with (
        patch.object(service, "_load_async", AsyncMock(return_value=config)),
        patch.object(service, "_management_async", AsyncMock()),
        patch.object(config, "preflight_reinitialization_async", AsyncMock(return_value=prepared)),
        patch.object(config, "apply_prepared_reinitialization_async", AsyncMock()),
        patch.object(lifecycle_module, "validate_reinitialization_memory"),
        patch.object(lifecycle_module, "resolve_environment_async", AsyncMock(return_value={"NEW_VALUE": "new"})),
        patch.object(lifecycle_module, "close_services_async", AsyncMock()),
        patch.object(lifecycle_module, "peek_scenario_run_service", return_value=None),
        patch.object(lifecycle_module, "outstanding_estimates", return_value=0),
    ):
        yield service


async def apply_async(runtime: RuntimeLifecycle) -> None:
    _, version = await runtime.source.read_with_version_async()
    result = runtime.begin_apply(version=version)
    assert result["outcome"] == "accepted"
    assert runtime.apply_task is not None
    await runtime.apply_task


@pytest.mark.parametrize("failure", [None, "scenarios", "services"])
async def test_shutdown_disposes_memory_after_services_even_on_failure(
    runtime: RuntimeLifecycle, failure: str | None
) -> None:
    memory = MagicMock(spec=MemoryInterface)
    scenario_service = MagicMock(spec=ScenarioRunService)
    order = MagicMock()
    with (
        patch.object(CentralMemory, "_memory_instance", memory),
        patch.object(lifecycle_module, "peek_scenario_run_service", return_value=scenario_service),
        patch.object(lifecycle_module, "close_services_async", new_callable=AsyncMock) as close_services,
    ):
        order.attach_mock(scenario_service.shutdown_async, "scenarios")
        order.attach_mock(close_services, "services")
        order.attach_mock(memory.dispose_engine_async, "memory")
        if failure is not None:
            getattr(order, failure).side_effect = RuntimeError("cleanup failed")
        with pytest.raises(RuntimeError, match="cleanup failed") if failure else nullcontext():
            await runtime.shutdown_async()

    assert order.mock_calls == [call.scenarios(), call.services(), call.memory()]


async def test_success_preflights_then_replaces_idle_runtime(runtime: RuntimeLifecycle) -> None:
    await apply_async(runtime)
    assert runtime.state == "ready"
    assert runtime.generation != "original"
    config = await runtime._load_async()
    config.preflight_reinitialization_async.assert_awaited_once_with(environment_values={"NEW_VALUE": "new"})
    lifecycle_module.close_services_async.assert_awaited_once()
    config.apply_prepared_reinitialization_async.assert_awaited_once()
    assert runtime.version == (await runtime.source.read_with_version_async())[1]


async def test_saved_opt_in_is_required(runtime: RuntimeLifecycle) -> None:
    config = await runtime._load_async()
    config.enable_live_reinitialization = False
    await apply_async(runtime)
    assert runtime.outcome == "unsupported"
    assert runtime.state == "ready"
    config.preflight_reinitialization_async.assert_not_awaited()
    lifecycle_module.close_services_async.assert_not_awaited()


async def test_stale_version_and_invalid_preflight_do_not_mutate(runtime: RuntimeLifecycle) -> None:
    runtime.begin_apply(version="old")
    assert runtime.apply_task is not None
    await runtime.apply_task
    assert runtime.outcome == "version-conflict"
    with patch.object(runtime, "_load_async", AsyncMock(side_effect=ValueError("secret"))):
        await apply_async(runtime)
    assert runtime.outcome == "invalid-configuration"
    assert runtime.state == "ready"
    assert "secret" not in runtime.message
    lifecycle_module.close_services_async.assert_not_awaited()


async def test_active_work_rejects_apply_without_stopping_or_mutating(runtime: RuntimeLifecycle) -> None:
    service = MagicMock()
    service.has_active_work.return_value = True
    with patch.object(lifecycle_module, "peek_scenario_run_service", return_value=service):
        await apply_async(runtime)
    assert runtime.outcome == "busy"
    assert runtime.state == "ready"
    lifecycle_module.close_services_async.assert_not_awaited()
    assert not hasattr(service, "request_stop") or not service.request_stop.called


async def test_admitted_manual_send_blocks_reinitialization_without_an_http_request_async(
    runtime: RuntimeLifecycle,
) -> None:
    get_manual_send_scheduler.cache_clear()
    try:
        scheduler = get_manual_send_scheduler()
        with scheduler.reserve(conversation_id="accepted-send"):
            assert not runtime.operations
            await apply_async(runtime)
            assert runtime.outcome == "busy"
            assert runtime.generation == "original"
            lifecycle_module.close_services_async.assert_not_awaited()
        await apply_async(runtime)
        assert runtime.generation != "original"
    finally:
        get_manual_send_scheduler.cache_clear()


async def test_second_idle_check_closes_admission_race(runtime: RuntimeLifecycle) -> None:
    with patch.object(runtime, "_has_active_work", side_effect=[False, True]):
        await apply_async(runtime)
    assert runtime.outcome == "busy"
    assert runtime.state == "ready"
    lifecycle_module.close_services_async.assert_not_awaited()


async def test_mutation_failure_requires_restart_and_cannot_retry(runtime: RuntimeLifecycle) -> None:
    config = await runtime._load_async()
    config.apply_prepared_reinitialization_async.side_effect = ValueError("credential=secret")
    await apply_async(runtime)
    assert runtime.state == "restart-required"
    assert runtime.outcome == "restart-required"
    assert "secret" not in runtime.message
    _, version = await runtime.source.read_with_version_async()
    assert runtime.begin_apply(version=version)["outcome"] == "restart-required"


async def test_apply_survives_client_and_concurrent_apply_is_busy(runtime: RuntimeLifecycle) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    config = await runtime._load_async()

    async def initialize_async(**kwargs: object) -> None:
        entered.set()
        await release.wait()

    config.apply_prepared_reinitialization_async.side_effect = initialize_async
    _, version = await runtime.source.read_with_version_async()
    runtime.begin_apply(version=version)
    await entered.wait()
    assert runtime.begin_apply(version=version)["outcome"] == "busy"
    assert runtime.state == "initializing"
    release.set()
    assert runtime.apply_task is not None
    await runtime.apply_task
    assert runtime.state == "ready"


async def test_disconnected_send_remains_owned_and_blocks_apply(runtime: RuntimeLifecycle) -> None:
    entered, release, persisted = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @runtime.app.post("/api/attacks/id/messages")
    async def send_async() -> dict[str, bool]:
        entered.set()
        await release.wait()
        persisted.set()
        return {"stored": True}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.app), base_url="http://test") as client:
        request = asyncio.create_task(client.post("/api/attacks/id/messages"))
        await entered.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert len(runtime.operations) == 1
        await apply_async(runtime)
        assert runtime.outcome == "busy"
        assert runtime.state == "ready"
        release.set()
        await persisted.wait()
        await asyncio.gather(*runtime.operations)
        await apply_async(runtime)
        assert runtime.state == "ready"


async def test_background_estimates_reject_apply(runtime: RuntimeLifecycle) -> None:
    with patch.object(lifecycle_module, "outstanding_estimates", return_value=2):
        await apply_async(runtime)
    assert runtime.state == "ready"
    assert runtime.outcome == "busy"
    lifecycle_module.close_services_async.assert_not_awaited()


async def test_apply_denies_writes_but_allows_repair_reads(runtime: RuntimeLifecycle) -> None:
    async with runtime.edit_lock:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.app), base_url="http://test") as client:
            assert (await client.put("/api/config", json={})).status_code == 503
            assert (await client.get("/api/config")).status_code == 200
            assert runtime.begin_apply(version="v")["outcome"] == "busy"


async def test_non_admin_apply_and_status_remain_denied(runtime: RuntimeLifecycle) -> None:
    runtime.app.dependency_overrides.clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.app), base_url="http://test") as client:
        assert (await client.get("/api/config/runtime")).status_code == 403
        assert (await client.post("/api/config/runtime/apply", json={"version": "v"})).status_code == 403


@pytest.mark.parametrize(
    "worker_setting",
    ["WEB_CONCURRENCY", "UVICORN_WORKERS", "PYRIT_API_WORKERS", "PYRIT_REPLICAS"],
)
def test_multiple_workers_disable_apply(worker_setting: str) -> None:
    with patch.dict(os.environ, {worker_setting: "2"}, clear=True):
        service = RuntimeLifecycle(app=FastAPI(), source=ConfigurationFileService(config_file_value=None))
    assert not service.topology_supported


async def test_invalid_cold_configuration_requires_restart_but_retains_raw_repair(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("broken: [", encoding="utf-8")
    source = ConfigurationFileService(config_file_value=str(path))
    app = FastAPI()
    service = RuntimeLifecycle(app=app, source=source)
    await service.startup_async()
    assert service.state == "restart-required"
    assert not app.state.allow_custom_initializers
    assert app.state.environment_file_service is None
    content, version = await source.read_with_version_async()
    assert content == "broken: ["
    await source.update_async("memory_db_type: in_memory\n", expected_version=version)
    assert "in_memory" in await source.read_async()


def test_authorization_policy_does_not_change_with_environment(runtime: RuntimeLifecycle) -> None:
    runtime.app.state.auth_environment["PYRIT_ALLOW_UNAUTHENTICATED_ADMIN"] = ""
    request = Request({"type": "http", "app": runtime.app})
    with patch.dict(os.environ, {"PYRIT_ALLOW_UNAUTHENTICATED_ADMIN": "true"}):
        with pytest.raises(Exception) as error:
            require_admin(request)
    assert error.value.status_code == 403


@pytest.mark.parametrize("enabled", [False, True])
async def test_startup_installs_only_explicit_local_job_port_and_starts_consumer_after_ready_async(
    *, runtime: RuntimeLifecycle, tmp_path: Path, enabled: bool
) -> None:
    config = await runtime._load_async()
    settings = LocalEvaluationJobSettings(
        root=tmp_path / "jobs", allowed_actor_ids=frozenset({"00000000-0000-4000-8000-000000000001"})
    )
    port = MagicMock(spec=LocalEvaluationJobPort)
    memory = MagicMock(spec=MemoryInterface)
    scenario_service = MagicMock(spec=ScenarioRunService)

    def start_consumer() -> None:
        assert runtime.state == "ready"
        assert runtime.app.state.evaluation_job_port is port

    port.start_consumer.side_effect = start_consumer
    with (
        patch.object(CohostPreflight, "from_environment_async", new_callable=AsyncMock, return_value=None),
        patch.object(LocalEvaluationJobSettings, "from_environment", return_value=settings if enabled else None),
        patch.object(
            LocalEvaluationJobSettings, "create_port_async", new_callable=AsyncMock, return_value=port
        ) as create,
        patch.object(config, "initialize_pyrit_async", new_callable=AsyncMock) as initialize,
        patch.object(CentralMemory, "get_memory_instance", return_value=memory),
        patch.object(lifecycle_module, "get_scenario_run_service", return_value=scenario_service),
    ):
        await runtime.startup_async()
    assert runtime.state == "ready"
    initialize.assert_awaited_once()
    scenario_service.reconcile_interrupted_runs_async.assert_awaited_once()
    if enabled:
        create.assert_awaited_once()
        assert create.call_args.kwargs["memory"] is memory
        port.startup_async.assert_awaited_once()
        port.start_consumer.assert_called_once()
        assert runtime.evaluation_jobs is port
        assert runtime.begin_apply(version="ignored")["outcome"] == "unsupported"
    else:
        create.assert_not_awaited()
        port.start_consumer.assert_not_called()
        assert runtime.evaluation_jobs is None and runtime.app.state.evaluation_job_port is None


@pytest.mark.parametrize("conflict", ["original_preview", "multiple_workers"])
async def test_local_job_startup_refuses_preview_and_topology_before_initialization_async(
    *, runtime: RuntimeLifecycle, tmp_path: Path, conflict: str
) -> None:
    config = await runtime._load_async()
    settings = LocalEvaluationJobSettings(
        root=tmp_path / "jobs", allowed_actor_ids=frozenset({"00000000-0000-4000-8000-000000000001"})
    )
    runtime.topology_supported = conflict != "multiple_workers"
    preflight = MagicMock(spec=CohostPreflight) if conflict == "original_preview" else None
    with (
        patch.object(CohostPreflight, "from_environment_async", new_callable=AsyncMock, return_value=preflight),
        patch.object(LocalEvaluationJobSettings, "from_environment", return_value=settings),
        patch.object(LocalEvaluationJobSettings, "create_port_async", new_callable=AsyncMock) as create,
        patch.object(config, "initialize_pyrit_async", new_callable=AsyncMock) as initialize,
    ):
        await runtime.startup_async()
    assert runtime.state == "restart-required" and runtime.outcome == "initialization-failed"
    assert runtime.app.state.evaluation_job_port is None and runtime.evaluation_jobs is None
    initialize.assert_not_awaited()
    create.assert_not_awaited()


async def test_local_job_startup_failure_closes_owned_port_without_publishing_readiness_async(
    *, runtime: RuntimeLifecycle, tmp_path: Path
) -> None:
    config = await runtime._load_async()
    settings = LocalEvaluationJobSettings(
        root=tmp_path / "jobs", allowed_actor_ids=frozenset({"00000000-0000-4000-8000-000000000001"})
    )
    port = MagicMock(spec=LocalEvaluationJobPort)
    port.startup_async.side_effect = ValueError("fixture startup refusal")
    with (
        patch.object(CohostPreflight, "from_environment_async", new_callable=AsyncMock, return_value=None),
        patch.object(LocalEvaluationJobSettings, "from_environment", return_value=settings),
        patch.object(LocalEvaluationJobSettings, "create_port_async", new_callable=AsyncMock, return_value=port),
        patch.object(config, "initialize_pyrit_async", new_callable=AsyncMock),
        patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock(spec=MemoryInterface)),
    ):
        await runtime.startup_async()
    assert runtime.state == "restart-required"
    assert runtime.evaluation_jobs is None and runtime.app.state.evaluation_job_port is None
    port.shutdown_async.assert_awaited_once()
    port.start_consumer.assert_not_called()


async def test_shutdown_keeps_canonical_memory_alive_until_local_job_writer_joins_async(
    runtime: RuntimeLifecycle,
) -> None:
    port = MagicMock(spec=LocalEvaluationJobPort)
    memory = MagicMock(spec=MemoryInterface)
    entered, release = asyncio.Event(), asyncio.Event()
    order: list[str] = []

    async def join_jobs_async() -> None:
        entered.set()
        await release.wait()
        order.append("jobs")

    async def close_async() -> None:
        order.append("services")

    async def dispose_async() -> None:
        order.append("memory")

    port.shutdown_async.side_effect = join_jobs_async
    memory.dispose_engine_async.side_effect = dispose_async
    runtime.evaluation_jobs = port
    runtime.app.state.evaluation_job_port = port
    with (
        patch.object(CentralMemory, "_memory_instance", memory),
        patch.object(lifecycle_module, "close_services_async", side_effect=close_async),
    ):
        caller = asyncio.create_task(runtime.shutdown_async())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert not caller.done() and order == []
            memory.dispose_engine_async.assert_not_awaited()
            release.set()
            await caller
        finally:
            release.set()
            await asyncio.gather(caller, return_exceptions=True)
    assert order == ["jobs", "services", "memory"]
    assert runtime.evaluation_jobs is None and runtime.app.state.evaluation_job_port is None
