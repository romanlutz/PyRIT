# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
from typing import Generic, TypeVar
from unittest.mock import AsyncMock, patch

import pytest

from pyrit.executor.workflow.environment_lease import EnvironmentCleanupError, EnvironmentLease
from pyrit.models.environment_lease import (
    EnvironmentCapability,
    EnvironmentCleanupStatus,
    EnvironmentHealth,
    EnvironmentLeaseSnapshot,
    EnvironmentResourceHandle,
    EnvironmentServiceHandle,
)

RuntimeT = TypeVar("RuntimeT")

ONE_SERVICE = (("sandbox", "agent", None),)
TWO_SERVICES = (("sandbox", "agent", None), ("web", "target", None))
FOUR_SERVICES = (
    ("control", "controller", None),
    ("worker", "agent", "control"),
    ("database", "target", "control"),
    ("api", "target", "database"),
)


class InertLease(EnvironmentLease[RuntimeT], Generic[RuntimeT]):
    """Only Python collections and events; no external controller, VM, process or network."""

    def __init__(
        self,
        *,
        run_id: str,
        runtime: RuntimeT,
        services: tuple[tuple[str, str, str | None], ...] = ONE_SERVICE,
        capabilities: frozenset[EnvironmentCapability] = frozenset(
            {EnvironmentCapability.SETUP, EnvironmentCapability.HEALTH_CHECK}
        ),
        required: frozenset[EnvironmentCapability] = frozenset(),
        order: list[str] | None = None,
        fail_acquire: str | None = None,
        fail_release: str | None = None,
        cleanup_timeout_seconds: float = 1,
    ) -> None:
        super().__init__(
            run_id=run_id,
            capabilities=capabilities,
            required_capabilities=required,
            cleanup_timeout_seconds=cleanup_timeout_seconds,
        )
        self.runtime = runtime
        self.specs = services
        self.order = order if order is not None else []
        self.live = {"externally-owned-controller", "another-runs-resource"}
        self.fail_acquire = fail_acquire
        self.fail_release = fail_release

    async def _acquire_async(self) -> RuntimeT:
        for name, role, parent in self.specs:
            handle = EnvironmentResourceHandle(
                run_id=self.run_id,
                provider="inert-external-controller",
                resource_id=f"{self.run_id}:{name}",
                kind="lease",
            )
            await self._acquire_resource_async(
                handle=handle, acquire_async=self._allocate_async, release_async=self._release_async
            )
            self._register_service(
                EnvironmentServiceHandle(name=name, roles=frozenset({role}), resource=handle, parent_name=parent)
            )
        return self.runtime

    async def _allocate_async(self, handle: EnvironmentResourceHandle) -> None:
        self.order.append(f"acquire:{handle.resource_id}")
        self.live.add(handle.resource_id)
        if handle.resource_id == self.fail_acquire:
            raise OSError("Inert acquisition failed after allocation.")

    async def _release_async(self, handle: EnvironmentResourceHandle) -> None:
        assert handle.run_id == self.run_id and handle.provider == "inert-external-controller"
        self.order.append(f"release:{handle.resource_id}")
        if handle.resource_id == self.fail_release:
            raise OSError("Inert controller could not confirm release.")
        self.live.discard(handle.resource_id)
        assert handle.resource_id not in self.live

    async def _setup_async(self) -> None:
        self.order.append("setup")

    async def _check_health_async(self) -> tuple[EnvironmentHealth, ...]:
        self.order.append("health")
        return tuple(
            EnvironmentHealth(service_name=service.name, healthy=True, reason="Inert provider observation.")
            for service in self.snapshot().services
        )


@pytest.mark.parametrize("services", [ONE_SERVICE, TWO_SERVICES, FOUR_SERVICES])
async def test_named_topologies_setup_health_and_reverse_owned_cleanup_async(
    services: tuple[tuple[str, str, str | None], ...],
) -> None:
    lease = InertLease(run_id="run-1", runtime="inert runtime", services=services)
    assert await lease.acquire_async() == "inert runtime"
    assert lease.cleanup_budget_seconds == len(services)
    snapshot = lease.snapshot()
    assert snapshot.state == "ready" and snapshot.setup_completed
    assert len(snapshot.health) == len(snapshot.services) == len(snapshot.resources) == len(services)
    assert all(item.acquisition_completed for item in snapshot.resources)
    for name, role, parent in services:
        service = lease.service(name)
        assert service.roles == frozenset({role}) and service.parent_name == parent
        assert service.resource.run_id == "run-1"
    assert EnvironmentLeaseSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot
    await lease.close_async()
    await lease.close_async()
    names = [f"run-1:{name}" for name, _, _ in services]
    assert lease.order == [
        *(f"acquire:{name}" for name in names),
        "setup",
        "health",
        *(f"release:{name}" for name in reversed(names)),
    ]
    assert lease.live == {"externally-owned-controller", "another-runs-resource"}
    assert all(item.status == "confirmed" for item in lease.snapshot().resources)
    assert snapshot.state == "ready" and all(item.status == "pending" for item in snapshot.resources)
    with pytest.raises(RuntimeError, match="ready"):
        lease.service(services[0][0])
    with pytest.raises(RuntimeError, match="only once"):
        await lease.acquire_async()


async def test_partial_acquire_rolls_back_including_the_failing_resource_async() -> None:
    lease = InertLease(run_id="run", runtime=None, services=FOUR_SERVICES, fail_acquire="run:database")
    with pytest.raises(OSError, match="after allocation"):
        await lease.acquire_async()
    assert lease.snapshot().state == "closed"
    assert [item.acquisition_completed for item in lease.snapshot().resources] == [True, True, False]
    assert lease.order == [
        "acquire:run:control",
        "acquire:run:worker",
        "acquire:run:database",
        "release:run:database",
        "release:run:worker",
        "release:run:control",
    ]
    assert lease.live == {"externally-owned-controller", "another-runs-resource"}
    assert "setup" not in lease.order and "health" not in lease.order
    await lease.close_async()
    assert len(lease.order) == 6


async def test_acquisition_cancellation_rolls_back_before_propagation_async() -> None:
    lease = InertLease(run_id="run", runtime=None, services=TWO_SERVICES)
    entered = asyncio.Event()
    allocate = lease._allocate_async

    async def wait_after_allocate_async(handle: EnvironmentResourceHandle) -> None:
        await allocate(handle)
        if handle.resource_id == "run:web":
            entered.set()
            await asyncio.Event().wait()

    with patch.object(lease, "_allocate_async", side_effect=wait_after_allocate_async):
        task = asyncio.create_task(lease.acquire_async())
        await asyncio.wait_for(entered.wait(), timeout=2)
        with pytest.raises(RuntimeError, match="Cancel and await"):
            await lease.close_async()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert lease.snapshot().state == "closed"
    assert lease.live == {"externally-owned-controller", "another-runs-resource"}
    assert lease.order[-2:] == ["release:run:web", "release:run:sandbox"]


async def test_cancellation_during_failed_acquire_rollback_remains_cancellation_async() -> None:
    lease = InertLease(run_id="run", runtime=None, fail_acquire="run:sandbox")
    entered, resume = asyncio.Event(), asyncio.Event()
    release = lease._release_async

    async def paused_release_async(handle: EnvironmentResourceHandle) -> None:
        entered.set()
        await resume.wait()
        await release(handle)

    with patch.object(lease, "_release_async", side_effect=paused_release_async):
        task = asyncio.create_task(lease.acquire_async())
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        resume.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
    assert isinstance(caught.value.__cause__, OSError)
    assert lease.snapshot().state == "closed"
    assert lease.live == {"externally-owned-controller", "another-runs-resource"}


async def test_cleanup_cancellation_does_not_interrupt_or_repeat_owned_release_async() -> None:
    lease = InertLease(run_id="run", runtime=None, services=TWO_SERVICES)
    await lease.acquire_async()
    entered, resume = asyncio.Event(), asyncio.Event()
    release = lease._release_async

    async def paused_release_async(handle: EnvironmentResourceHandle) -> None:
        entered.set()
        await resume.wait()
        await release(handle)

    for resource in lease._resources.values():
        resource.release_async = paused_release_async
    task = asyncio.create_task(lease.close_async())
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    resume.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await lease.close_async()
    assert lease.order[-2:] == ["release:run:web", "release:run:sandbox"]
    assert lease.live == {"externally-owned-controller", "another-runs-resource"}


async def test_failed_release_attempts_all_resources_and_never_reports_clean_success_async() -> None:
    lease = InertLease(run_id="run", runtime=None, services=FOUR_SERVICES, fail_release="run:database")
    await lease.acquire_async()
    for _ in range(2):
        with pytest.raises(EnvironmentCleanupError, match="not confirmed released"):
            await lease.close_async()
    assert lease.order[-4:] == ["release:run:api", "release:run:database", "release:run:worker", "release:run:control"]
    assert lease.snapshot().state == "cleanup_failed"
    assert [item.status for item in lease.snapshot().resources] == ["confirmed", "confirmed", "failed", "confirmed"]
    assert lease.live == {"externally-owned-controller", "another-runs-resource", "run:database"}


async def test_failed_acquisition_preserves_failed_rollback_as_cause_async() -> None:
    lease = InertLease(run_id="run", runtime=None, fail_acquire="run:sandbox", fail_release="run:sandbox")
    with pytest.raises(OSError, match="after allocation") as caught:
        await lease.acquire_async()
    assert isinstance(caught.value.__cause__, EnvironmentCleanupError)
    assert lease.snapshot().state == "cleanup_failed"
    assert any("Acquisition failed" in error for error in lease.snapshot().errors)
    assert any("Cleanup failed" in error for error in lease.snapshot().errors)


async def test_cleanup_timeout_is_explicit_and_remaining_resources_are_released_async() -> None:
    lease = InertLease(run_id="run", runtime=None, services=TWO_SERVICES, cleanup_timeout_seconds=0.01)
    release = lease._release_async

    async def stuck_release_async(handle: EnvironmentResourceHandle) -> None:
        if handle.resource_id == "run:web":
            await asyncio.Event().wait()
        await release(handle)

    with patch.object(lease, "_release_async", side_effect=stuck_release_async):
        await lease.acquire_async()
    with pytest.raises(EnvironmentCleanupError):
        await lease.close_async()
    assert lease.snapshot().resources[1].status is EnvironmentCleanupStatus.FAILED
    assert "TimeoutError" in lease.snapshot().resources[1].error
    assert "run:sandbox" not in lease.live and "run:web" in lease.live


@pytest.mark.parametrize("phase", ["setup", "health"])
async def test_setup_and_health_failure_trigger_rollback_async(phase: str) -> None:
    lease = InertLease(run_id="run", runtime=None)
    method = "_setup_async" if phase == "setup" else "_check_health_async"
    with patch.object(lease, method, new_callable=AsyncMock, side_effect=OSError("Inert provider operation failed.")):
        with pytest.raises(OSError, match="operation failed"):
            await lease.acquire_async()
    assert lease.snapshot().state == "closed"
    assert lease.live == {"externally-owned-controller", "another-runs-resource"}


@pytest.mark.parametrize(
    "health",
    [
        (),
        (EnvironmentHealth(service_name="other", healthy=True, reason="Wrong identity"),),
        (EnvironmentHealth(service_name="sandbox", healthy=False, reason="Unavailable"),),
        (EnvironmentHealth(service_name="sandbox", healthy=None, reason="Unobserved"),),
        (EnvironmentHealth(service_name="sandbox", healthy=True, reason="Repeated"),) * 2,
    ],
)
async def test_health_requires_complete_observations_not_inferred_success_async(
    health: tuple[EnvironmentHealth, ...],
) -> None:
    lease = InertLease(run_id="run", runtime=None)
    with patch.object(lease, "_check_health_async", new_callable=AsyncMock, return_value=health):
        with pytest.raises(RuntimeError, match="health"):
            await lease.acquire_async()
    assert lease.snapshot().health == health and lease.snapshot().state == "closed"


@pytest.mark.parametrize("required", list(EnvironmentCapability))
async def test_unsupported_required_capability_fails_before_acquisition_async(required: EnvironmentCapability) -> None:
    lease = InertLease(run_id="run", runtime=None, capabilities=frozenset(), required=frozenset({required}))
    with pytest.raises(NotImplementedError, match="Unsupported"):
        await lease.acquire_async()
    assert lease.order == [] and lease.snapshot().state == "new"


@pytest.mark.parametrize(
    "future", [EnvironmentCapability.DYNAMIC_SERVICES, EnvironmentCapability.NETWORK_PHASE_TRANSITIONS]
)
def test_unimplemented_future_operation_cannot_be_advertised(future: EnvironmentCapability) -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        InertLease(run_id="run", runtime=None, capabilities=frozenset({future}))


async def test_unsupported_optional_operations_are_not_silent_noops_async() -> None:
    lease = InertLease(run_id="run", runtime=None, capabilities=frozenset())
    await lease.acquire_async()
    with pytest.raises(NotImplementedError):
        await lease.setup_async()
    with pytest.raises(NotImplementedError):
        await lease.check_health_async()
    assert not lease.snapshot().setup_completed and lease.snapshot().health == ()
    await lease.close_async()


@pytest.mark.parametrize("capability", [EnvironmentCapability.SETUP, EnvironmentCapability.HEALTH_CHECK])
def test_unimplemented_provider_hook_cannot_be_advertised(capability: EnvironmentCapability) -> None:
    class MissingHooks(EnvironmentLease[None]):
        async def _acquire_async(self) -> None:
            raise AssertionError("No acquisition should be attempted.")

    with pytest.raises(NotImplementedError, match="requires a provider implementation"):
        MissingHooks(run_id="run", capabilities=frozenset({capability}))


async def test_concurrent_health_check_and_cleanup_are_rejected_async() -> None:
    lease = InertLease(run_id="run", runtime=None)
    await lease.acquire_async()
    entered = asyncio.Event()

    async def paused_health_async() -> tuple[EnvironmentHealth, ...]:
        entered.set()
        await asyncio.Event().wait()
        return ()

    with patch.object(lease, "_check_health_async", side_effect=paused_health_async):
        task = asyncio.create_task(lease.check_health_async())
        await asyncio.wait_for(entered.wait(), timeout=2)
        with pytest.raises(RuntimeError, match="already running"):
            await lease.check_health_async()
        with pytest.raises(RuntimeError, match="active health"):
            await lease.close_async()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await lease.close_async()
    assert lease.live == {"externally-owned-controller", "another-runs-resource"}
    assert lease.snapshot().health == ()


async def test_closed_unused_lease_cannot_race_a_new_acquisition_async() -> None:
    lease = InertLease(run_id="run", runtime=None)
    await lease.close_async()
    with pytest.raises(RuntimeError, match="only once"):
        await lease.acquire_async()
    assert lease.snapshot().state == "closed" and lease.order == []


async def test_untracked_acquisition_task_cannot_allocate_resources_async() -> None:
    lease = InertLease(run_id="run", runtime=None)
    acquire = lease._acquire_async

    async def spawned_acquire_async() -> None:
        task = asyncio.create_task(acquire())
        with pytest.raises(RuntimeError, match="acquiring owner task"):
            await task
        await acquire()

    with patch.object(lease, "_acquire_async", side_effect=spawned_acquire_async):
        await lease.acquire_async()
    await lease.close_async()
    assert lease.order.count("acquire:run:sandbox") == lease.order.count("release:run:sandbox") == 1


async def test_foreign_or_duplicate_resource_never_gets_another_cleanup_owner_async() -> None:
    lease = InertLease(run_id="run", runtime=None)

    async def invalid_acquire_async() -> None:
        handle = EnvironmentResourceHandle(run_id="other", provider="p", resource_id="external", kind="allocation")
        acquire, release = AsyncMock(), AsyncMock()
        with pytest.raises(ValueError, match="another run"):
            await lease._acquire_resource_async(handle=handle, acquire_async=acquire, release_async=release)
        acquire.assert_not_awaited()
        release.assert_not_awaited()
        valid = handle.model_copy(update={"run_id": "run"})
        await lease._acquire_resource_async(handle=valid, acquire_async=acquire, release_async=release)
        with pytest.raises(ValueError, match="one cleanup owner"):
            await lease._acquire_resource_async(
                handle=valid.model_copy(update={"kind": "different"}), acquire_async=acquire, release_async=release
            )
        lease._register_service(
            EnvironmentServiceHandle(name="external", roles=frozenset({"controller"}), resource=valid)
        )

    with patch.object(lease, "_acquire_async", side_effect=invalid_acquire_async):
        await lease.acquire_async()
    await lease.close_async()
    assert len(lease.snapshot().resources) == 1


async def test_multiple_named_services_can_share_one_external_allocation_async() -> None:
    lease = InertLease(run_id="run", runtime=None)
    acquire = lease._acquire_async

    async def shared_acquire_async() -> None:
        await acquire()
        resource = lease.snapshot().resources[0].resource
        lease._register_service(
            EnvironmentServiceHandle(
                name="guest", roles=frozenset({"target", "worker"}), parent_name="sandbox", resource=resource
            )
        )

    with patch.object(lease, "_acquire_async", side_effect=shared_acquire_async):
        await lease.acquire_async()
    assert len(lease.snapshot().services) == 2 and len(lease.snapshot().resources) == 1
    await lease.close_async()
    assert lease.order.count("release:run:sandbox") == 1


async def test_service_names_and_parent_references_are_validated_async() -> None:
    lease = InertLease(run_id="run", runtime=None)
    acquire = lease._acquire_async

    async def invalid_services_async() -> None:
        await acquire()
        service = lease.snapshot().services[0]
        with pytest.raises(ValueError, match="unique name"):
            lease._register_service(service)
        with pytest.raises(ValueError, match="existing named parent"):
            lease._register_service(service.model_copy(update={"name": "child", "parent_name": "missing"}))
        with pytest.raises(ValueError, match="owned resource"):
            lease._register_service(
                service.model_copy(
                    update={"name": "foreign", "resource": service.resource.model_copy(update={"run_id": "other"})}
                )
            )

    with patch.object(lease, "_acquire_async", side_effect=invalid_services_async):
        await lease.acquire_async()
    await lease.close_async()
