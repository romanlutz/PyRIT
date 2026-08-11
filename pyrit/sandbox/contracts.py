# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Provider, session, environment, and process sandbox contracts."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from pyrit.models import ComponentIdentifier, Identifiable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pyrit.executor.capability.evidence import CapabilityEvidenceSink
    from pyrit.sandbox.models import (
        SandboxConnectionInfo,
        SandboxExecRequest,
        SandboxExecResult,
        SandboxOperationContext,
        SandboxReadResult,
        SandboxSessionSpec,
        SandboxTaskSpec,
        SandboxWriteResult,
    )


async def _await_cleanup_task_async(*, cleanup_task: asyncio.Task[None]) -> None:
    """Finish one cleanup task despite repeated caller cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
    try:
        cleanup_task.result()
    except BaseException as cleanup_error:
        if cancellation is not None:
            raise cleanup_error from cancellation
        raise
    if cancellation is not None:
        raise cancellation


class SandboxProcess(ABC):
    """A future-compatible seam for streaming and buffered process execution."""

    @abstractmethod
    async def communicate_async(
        self,
        *,
        stdin: bytes | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> SandboxExecResult:
        """Wait for completion and return bounded buffered output."""

    @abstractmethod
    async def terminate_async(self) -> None:
        """Terminate the process and its descendants."""


class SandboxEnvironment(ABC):
    """A named execution and filesystem surface owned by one session."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The environment name."""

    @property
    @abstractmethod
    def connection_info(self) -> SandboxConnectionInfo:
        """Non-secret connection metadata."""

    @abstractmethod
    async def start_process_async(
        self,
        *,
        request: SandboxExecRequest,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxProcess:
        """Start a process without waiting for buffered completion."""

    @abstractmethod
    async def exec_async(
        self,
        *,
        request: SandboxExecRequest,
        cancellation_event: asyncio.Event | None = None,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxExecResult:
        """Execute a bounded buffered process request."""

    @abstractmethod
    async def read_file_async(
        self,
        *,
        path: str,
        max_bytes: int | None = None,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxReadResult:
        """Read a binary file relative to the environment root."""

    @abstractmethod
    async def write_file_async(
        self,
        *,
        path: str,
        data: bytes,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxWriteResult:
        """Write a binary file relative to the environment root."""


class SandboxSession(ABC):
    """A per-attempt owner of named environments and their cleanup."""

    def __init__(
        self,
        *,
        provider_name: str,
        spec: SandboxSessionSpec,
        evidence_sink: CapabilityEvidenceSink | None = None,
    ) -> None:
        """Initialize session identity and cleanup state."""
        self._provider_name = provider_name
        self._spec = spec
        self._evidence_sink = evidence_sink
        self._lifecycle_lock = asyncio.Lock()
        self._initialized = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def session_id(self) -> str:
        """The session identity."""
        return self._spec.session_id

    @property
    def spec(self) -> SandboxSessionSpec:
        """The immutable session specification."""
        return self._spec

    @property
    @abstractmethod
    def environments(self) -> tuple[SandboxEnvironment, ...]:
        """The session environments."""

    def get_environment(self, name: str | None = None) -> SandboxEnvironment:
        """
        Resolve a named environment or the session's deterministic default.

        Returns:
            SandboxEnvironment: The resolved environment.

        Raises:
            KeyError: If the requested environment is not available.
        """
        resolved_name = name or self._spec.resolve_default_environment()
        for environment in self.environments:
            if environment.name == resolved_name:
                return environment
        available = ", ".join(sorted(environment.name for environment in self.environments))
        raise KeyError(f"Sandbox environment '{resolved_name}' is not available. Available: {available}")

    async def initialize_async(self) -> None:
        """
        Initialize all environments, cleaning the session on failure.

        Raises:
            RuntimeError: If the session was already closed.
        """
        async with self._lifecycle_lock:
            if self._initialized:
                return
            if self._close_task is not None:
                raise RuntimeError(f"Sandbox session '{self.session_id}' is already closed.")
            try:
                await self._initialize_async()
            except BaseException as error:
                self._close_task = asyncio.create_task(self._close_async())
                try:
                    await _await_cleanup_task_async(cleanup_task=self._close_task)
                except BaseException as cleanup_error:
                    raise cleanup_error from error
                raise
            self._initialized = True

    async def close_async(self) -> None:
        """Clean up the session exactly once."""
        async with self._lifecycle_lock:
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._close_async())
            cleanup_task = self._close_task
        await _await_cleanup_task_async(cleanup_task=cleanup_task)

    @abstractmethod
    async def _initialize_async(self) -> None:
        """Perform provider-specific environment setup."""

    @abstractmethod
    async def _close_async(self) -> None:
        """Perform provider-specific session cleanup."""

    async def __aenter__(self) -> SandboxSession:
        """
        Initialize and enter the session.

        Returns:
            SandboxSession: This initialized session.
        """
        await self.initialize_async()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Clean up the session."""
        await self.close_async()


class SandboxProvider(Identifiable, ABC):
    """Own provider-wide resources and create independent attempt sessions."""

    def __init__(self) -> None:
        """Initialize provider lifecycle state."""
        self._lifecycle_lock = asyncio.Lock()
        self._prepared = False
        self._cleanup_task: asyncio.Task[None] | None = None
        self._task_prepare_tasks: dict[str, asyncio.Task[None]] = {}
        self._task_cleanup_tasks: dict[str, asyncio.Task[None]] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """The stable provider name."""

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the provider identity.

        Returns:
            ComponentIdentifier: The provider class and stable name.
        """
        return ComponentIdentifier(
            class_name=type(self).__name__,
            class_module=type(self).__module__,
            params={"name": self.name},
        )

    async def prepare_async(self) -> None:
        """
        Prepare provider resources once, with cleanup after partial failure.

        Raises:
            RuntimeError: If the provider was already cleaned up.
        """
        async with self._lifecycle_lock:
            if self._prepared:
                return
            if self._cleanup_task is not None:
                raise RuntimeError(f"Sandbox provider '{self.name}' has already been cleaned up.")
            try:
                await self._prepare_async()
            except BaseException as error:
                self._cleanup_task = asyncio.create_task(self._cleanup_async())
                try:
                    await _await_cleanup_task_async(cleanup_task=self._cleanup_task)
                except BaseException as cleanup_error:
                    raise cleanup_error from error
                raise
            self._prepared = True

    async def cleanup_async(self) -> None:
        """Clean provider resources exactly once."""
        async with self._lifecycle_lock:
            if self._cleanup_task is None:
                self._prepared = False
                self._cleanup_task = asyncio.create_task(self._cleanup_async())
            cleanup_task = self._cleanup_task
        await _await_cleanup_task_async(cleanup_task=cleanup_task)

    async def prepare_task_async(self, task: SandboxTaskSpec) -> None:
        """
        Prepare task-scoped resources once.

        Raises:
            RuntimeError: If the provider is not prepared or cleanup has started.
        """
        async with self._lifecycle_lock:
            if not self._prepared or self._cleanup_task is not None:
                raise RuntimeError(f"Sandbox provider '{self.name}' is not available for task preparation.")
            preparation_task = self._task_prepare_tasks.get(task.task_id)
            if preparation_task is None:
                preparation_task = asyncio.create_task(self._prepare_task_async(task))
                self._task_prepare_tasks[task.task_id] = preparation_task
        try:
            await asyncio.shield(preparation_task)
        except BaseException as error:
            try:
                await self.cleanup_task_async(task)
            except BaseException as cleanup_error:
                raise cleanup_error from error
            raise

    async def cleanup_task_async(self, task: SandboxTaskSpec) -> None:
        """Clean task-scoped resources exactly once."""
        async with self._lifecycle_lock:
            cleanup_task = self._task_cleanup_tasks.get(task.task_id)
            if cleanup_task is None:
                cleanup_task = asyncio.create_task(self._cleanup_task_async(task))
                self._task_cleanup_tasks[task.task_id] = cleanup_task
        await _await_cleanup_task_async(cleanup_task=cleanup_task)

    async def create_session_async(
        self,
        *,
        spec: SandboxSessionSpec,
        evidence_sink: CapabilityEvidenceSink | None = None,
    ) -> SandboxSession:
        """
        Create one uninitialized per-attempt session.

        Returns:
            SandboxSession: The new uninitialized session.

        Raises:
            RuntimeError: If provider preparation has not completed.
        """
        async with self._lifecycle_lock:
            if not self._prepared or self._cleanup_task is not None:
                raise RuntimeError(
                    f"Sandbox provider '{self.name}' must be prepared and not cleaning up before creating sessions."
                )
            return await self._create_session_async(spec=spec, evidence_sink=evidence_sink)

    @asynccontextmanager
    async def managed_async(self) -> AsyncIterator[SandboxProvider]:
        """
        Prepare and clean provider resources around a block.

        Yields:
            SandboxProvider: This prepared provider.
        """
        await self.prepare_async()
        try:
            yield self
        finally:
            await self.cleanup_async()

    @asynccontextmanager
    async def task_async(self, task: SandboxTaskSpec) -> AsyncIterator[None]:
        """Prepare and clean task resources around a block."""
        await self.prepare_task_async(task)
        try:
            yield
        finally:
            await self.cleanup_task_async(task)

    @asynccontextmanager
    async def session_async(
        self,
        *,
        spec: SandboxSessionSpec,
        evidence_sink: CapabilityEvidenceSink | None = None,
    ) -> AsyncIterator[SandboxSession]:
        """
        Create, initialize, and clean one attempt session.

        Yields:
            SandboxSession: The initialized attempt session.
        """
        session = await self.create_session_async(spec=spec, evidence_sink=evidence_sink)
        async with session:
            yield session

    async def cleanup_orphans_async(self) -> int:
        """
        Clean abandoned provider resources.

        Returns:
            int: The number of resources cleaned.
        """
        return await self._cleanup_orphans_async()

    async def _prepare_async(self) -> None:
        """Optionally prepare provider-wide resources."""
        _ = self.name

    async def _prepare_task_async(self, task: SandboxTaskSpec) -> None:
        """Optionally prepare resources shared by task attempts."""
        _ = task

    async def _cleanup_task_async(self, task: SandboxTaskSpec) -> None:
        """Optionally clean resources shared by task attempts."""
        _ = task

    @abstractmethod
    async def _create_session_async(
        self,
        *,
        spec: SandboxSessionSpec,
        evidence_sink: CapabilityEvidenceSink | None,
    ) -> SandboxSession:
        """Create one provider-specific session."""

    @abstractmethod
    async def _cleanup_async(self) -> None:
        """Clean provider-wide resources."""

    @abstractmethod
    async def _cleanup_orphans_async(self) -> int:
        """Clean abandoned provider resources."""
