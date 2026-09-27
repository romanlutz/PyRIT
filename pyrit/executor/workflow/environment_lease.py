# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run-owned environment lifecycle without a dependency on any sandbox provider."""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar
from uuid import uuid4

from pyrit.models.environment_lease import (
    EnvironmentCapability,
    EnvironmentCleanupStatus,
    EnvironmentHealth,
    EnvironmentLeaseSnapshot,
    EnvironmentLeaseState,
    EnvironmentResourceCleanup,
    EnvironmentResourceHandle,
    EnvironmentServiceHandle,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

RuntimeT = TypeVar("RuntimeT")
ResourceT = TypeVar("ResourceT")


@dataclass
class _OwnedResource:
    handle: EnvironmentResourceHandle
    release_async: Callable[[EnvironmentResourceHandle], Awaitable[None]]
    acquisition_completed: bool = False
    status: EnvironmentCleanupStatus = EnvironmentCleanupStatus.PENDING
    error: str | None = None


class EnvironmentCleanupError(RuntimeError):
    """One or more owned resources could not be confirmed released."""


class EnvironmentLease(ABC, Generic[RuntimeT]):
    """Acquire a provider-defined topology and release only its run-owned resources."""

    def __init__(
        self,
        *,
        run_id: str,
        capabilities: frozenset[EnvironmentCapability] = frozenset(),
        required_capabilities: frozenset[EnvironmentCapability] = frozenset(),
        cleanup_timeout_seconds: float = 10,
    ) -> None:
        """
        Initialize an unused lease; construction performs no provider I/O.

        Args:
            run_id (str): The owning evaluation identity.
            capabilities (frozenset[EnvironmentCapability]): Provider-implemented operations.
            required_capabilities (frozenset[EnvironmentCapability]): Operations required before acquisition.
            cleanup_timeout_seconds (float): Cooperative deadline for each reserved resource's cleanup.

        Raises:
            ValueError: If identity or cleanup bounds are invalid.
            NotImplementedError: If a future capability has no implementation in this lease.
        """
        if not run_id.strip() or not math.isfinite(cleanup_timeout_seconds) or cleanup_timeout_seconds <= 0:
            raise ValueError("An environment lease requires a run identity and a positive finite cleanup deadline.")
        if capabilities - {EnvironmentCapability.SETUP, EnvironmentCapability.HEALTH_CHECK}:
            raise NotImplementedError(
                "Dynamic services and network-phase transitions are not implemented in this lease."
            )
        for capability, method in (
            (EnvironmentCapability.SETUP, "_setup_async"),
            (EnvironmentCapability.HEALTH_CHECK, "_check_health_async"),
        ):
            if capability in capabilities and getattr(type(self), method) is getattr(EnvironmentLease, method):
                raise NotImplementedError(f"{capability.value} requires a provider implementation.")
        self._run_id = run_id
        self._lease_id = str(uuid4())
        self._capabilities = frozenset(capabilities)
        self._required_capabilities = frozenset(required_capabilities)
        self._cleanup_timeout_seconds = cleanup_timeout_seconds
        self._state = EnvironmentLeaseState.NEW
        self._resources: dict[EnvironmentResourceHandle, _OwnedResource] = {}
        self._services: dict[str, EnvironmentServiceHandle] = {}
        self._setup_completed = False
        self._health: tuple[EnvironmentHealth, ...] = ()
        self._errors: list[str] = []
        self._close_task: asyncio.Task[None] | None = None
        self._acquire_task: asyncio.Task[object] | None = None
        self._health_task: asyncio.Task[object] | None = None

    @property
    def run_id(self) -> str:
        """The immutable owning evaluation identity."""
        return self._run_id

    @property
    def lease_id(self) -> str:
        """The immutable identity of this single-use lease."""
        return self._lease_id

    @property
    def capabilities(self) -> frozenset[EnvironmentCapability]:
        """The provider's declared and implemented optional operations."""
        return self._capabilities

    async def acquire_async(self) -> RuntimeT:
        """
        Acquire, optionally set up, and health-check before exposing the runtime.

        Returns:
            RuntimeT: The binding's runtime, never a canonical-model client object.

        Raises:
            RuntimeError: If reused, unhealthy, or missing service handles.
            NotImplementedError: If a required capability is unsupported.
            Exception: If acquisition or setup fails; registered resources are rolled back.
            asyncio.CancelledError: If cancelled; rollback still completes before propagation.
        """
        if self._state is not EnvironmentLeaseState.NEW:
            raise RuntimeError("Each environment lease may be acquired only once.")
        self.require_capabilities(self._required_capabilities)
        self._state = EnvironmentLeaseState.ACQUIRING
        self._acquire_task = asyncio.current_task()
        try:
            runtime = await self._acquire_async()
            if self._state is not EnvironmentLeaseState.ACQUIRING:
                raise RuntimeError("The provider closed the lease before acquisition completed.")
            if not self._services:
                raise RuntimeError("The acquired environment has no named service handles.")
            if EnvironmentCapability.SETUP in self.capabilities:
                await self.setup_async()
            if EnvironmentCapability.HEALTH_CHECK in self.capabilities:
                await self.check_health_async()
            self._state = EnvironmentLeaseState.READY
            return runtime
        except (Exception, asyncio.CancelledError) as error:
            self._errors.append(f"Acquisition failed: {type(error).__name__}: {error}")
            try:
                await self.close_async()
            except (Exception, asyncio.CancelledError) as cleanup_error:
                if isinstance(cleanup_error, asyncio.CancelledError):
                    raise cleanup_error from error
                raise error from cleanup_error
            raise
        finally:
            self._acquire_task = None

    async def setup_async(self) -> None:
        """
        Run provider setup once, without defining task prompts, grading, or network policy.

        Raises:
            NotImplementedError: If setup is not supported.
            RuntimeError: If the lease is inactive or setup was already completed.
        """
        self.require_capabilities(frozenset({EnvironmentCapability.SETUP}))
        self._require_active()
        if self._setup_completed:
            raise RuntimeError("Environment setup cannot be repeated.")
        await self._setup_async()
        self._setup_completed = True

    async def check_health_async(self) -> tuple[EnvironmentHealth, ...]:
        """
        Require one healthy provider observation for every named service.

        Returns:
            tuple[EnvironmentHealth, ...]: Observations retained in the lease snapshot.

        Raises:
            NotImplementedError: If health checks are not supported.
            RuntimeError: If the lease is inactive or observations are incomplete/unhealthy.
        """
        self.require_capabilities(frozenset({EnvironmentCapability.HEALTH_CHECK}))
        self._require_active()
        if self._health_task is not None:
            raise RuntimeError("A health check is already running.")
        self._health_task = asyncio.current_task()
        self._health = ()
        try:
            self._health = tuple(await self._check_health_async())
        finally:
            self._health_task = None
        names = [item.service_name for item in self._health]
        if len(names) != len(set(names)) or set(names) != set(self._services):
            raise RuntimeError("Environment health must cover each named service exactly once.")
        if any(item.healthy is not True for item in self._health):
            raise RuntimeError("One or more environment services are unhealthy or have unknown health.")
        return self._health

    def require_capabilities(self, required: frozenset[EnvironmentCapability]) -> None:
        """
        Reject unavailable capabilities before the caller attempts the operation.

        Raises:
            NotImplementedError: If any required operation is not supported.
        """
        missing = required - self.capabilities
        if missing:
            raise NotImplementedError(f"Unsupported environment capabilities: {', '.join(sorted(missing))}")

    def service(self, name: str) -> EnvironmentServiceHandle:
        """
        Resolve a named service only on an acquired, ready environment.

        Returns:
            EnvironmentServiceHandle: The exact registered handle.

        Raises:
            RuntimeError: If the lease is not ready.
            KeyError: If the requested service is not registered.
        """
        if self._state is not EnvironmentLeaseState.READY:
            raise RuntimeError("Service access requires a ready environment lease.")
        return self._services[name]

    def snapshot(self) -> EnvironmentLeaseSnapshot:
        """
        Describe observed lifecycle state without exposing provider clients or callbacks.

        Returns:
            EnvironmentLeaseSnapshot: Immutable identity, health and cleanup metadata.
        """
        return EnvironmentLeaseSnapshot(
            run_id=self.run_id,
            lease_id=self.lease_id,
            state=self._state,
            capabilities=self.capabilities,
            services=tuple(self._services.values()),
            resources=tuple(
                EnvironmentResourceCleanup(
                    resource=item.handle,
                    acquisition_completed=item.acquisition_completed,
                    status=item.status,
                    error=item.error,
                )
                for item in self._resources.values()
            ),
            setup_completed=self._setup_completed,
            health=self._health,
            errors=tuple(self._errors),
        )

    async def close_async(self) -> None:
        """
        Release resources in reverse reservation order once, including partial acquisitions.

        Caller cancellation does not interrupt release. A failed release is never
        retried implicitly, and all other registered resources are still attempted.

        Raises:
            RuntimeError: If another task is still acquiring or health-checking resources.
            EnvironmentCleanupError: If any release fails or exceeds its cooperative deadline.
            asyncio.CancelledError: If the caller cancels; cleanup is settled before propagation.
        """
        if self._acquire_task is not None and self._acquire_task is not asyncio.current_task():
            raise RuntimeError("Cancel and await acquisition rollback before closing from another task.")
        if self._health_task is not None:
            raise RuntimeError("Cancel and await the active health check before closing.")
        if self._close_task is None:
            self._state = EnvironmentLeaseState.CLOSING
            self._close_task = asyncio.create_task(self._release_resources_async())
        cancellation: asyncio.CancelledError | None = None
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except EnvironmentCleanupError:
                break
        try:
            self._close_task.result()
        except EnvironmentCleanupError as error:
            if cancellation is not None:
                raise cancellation from error
            raise
        if cancellation is not None:
            raise cancellation

    @abstractmethod
    async def _acquire_async(self) -> RuntimeT:
        """Build provider-owned services using the registered-resource acquisition helper."""
        ...

    async def _setup_async(self) -> None:
        raise NotImplementedError("This environment provider does not implement setup.")

    async def _check_health_async(self) -> tuple[EnvironmentHealth, ...]:
        raise NotImplementedError("This environment provider does not implement health checks.")

    async def _acquire_resource_async(
        self,
        *,
        handle: EnvironmentResourceHandle,
        acquire_async: Callable[[EnvironmentResourceHandle], Awaitable[ResourceT]],
        release_async: Callable[[EnvironmentResourceHandle], Awaitable[None]],
    ) -> ResourceT:
        self._require_acquiring_owner()
        if handle.run_id != self.run_id:
            raise ValueError("An environment lease cannot own another run's resource.")
        if any(item.provider == handle.provider and item.resource_id == handle.resource_id for item in self._resources):
            raise ValueError("An environment resource may have only one cleanup owner.")
        resource = _OwnedResource(handle=handle, release_async=release_async)
        self._resources[handle] = resource
        result = await acquire_async(handle)
        resource.acquisition_completed = True
        return result

    def _register_service(self, service: EnvironmentServiceHandle) -> None:
        self._require_acquiring_owner()
        if service.name in self._services or service.resource not in self._resources:
            raise ValueError("A service needs a unique name and an owned resource.")
        if not self._resources[service.resource].acquisition_completed:
            raise ValueError("A service cannot expose an unfinished resource acquisition.")
        if service.parent_name is not None and service.parent_name not in self._services:
            raise ValueError("A child service must reference an existing named parent.")
        self._services[service.name] = service

    def _require_acquiring_owner(self) -> None:
        if self._state is not EnvironmentLeaseState.ACQUIRING or self._acquire_task is not asyncio.current_task():
            raise RuntimeError(
                "Resources and services require the acquiring owner task; dynamic acquisition is unsupported."
            )

    def _require_active(self) -> None:
        if self._state not in {EnvironmentLeaseState.ACQUIRING, EnvironmentLeaseState.READY}:
            raise RuntimeError("The environment lease is not active.")
        if self._acquire_task is not None and self._acquire_task is not asyncio.current_task():
            raise RuntimeError("Only the acquiring task can operate on an unfinished lease.")

    async def _release_resources_async(self) -> None:
        self._state = EnvironmentLeaseState.CLOSING
        failures: list[BaseException] = []
        for resource in reversed(tuple(self._resources.values())):
            try:
                async with asyncio.timeout(self._cleanup_timeout_seconds):
                    await resource.release_async(resource.handle)
                resource.status = EnvironmentCleanupStatus.CONFIRMED
            except (Exception, asyncio.CancelledError) as error:
                resource.status = EnvironmentCleanupStatus.FAILED
                resource.error = f"{type(error).__name__}: {error}"
                self._errors.append(f"Cleanup failed for {resource.handle.resource_id}: {resource.error}")
                failures.append(error)
        self._state = EnvironmentLeaseState.CLEANUP_FAILED if failures else EnvironmentLeaseState.CLOSED
        if failures:
            raise EnvironmentCleanupError("One or more owned environment resources were not confirmed released.") from (
                BaseExceptionGroup("Environment release failures", failures)
            )
