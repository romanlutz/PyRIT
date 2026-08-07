# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Trusted local sandbox provider for development and tests."""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import os
import shutil
import signal
import stat
import subprocess
import tempfile
from contextlib import suppress
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles

from pyrit.executor.capability.models import SandboxOperationEvidence
from pyrit.sandbox.contracts import SandboxEnvironment, SandboxProcess, SandboxProvider, SandboxSession
from pyrit.sandbox.models import (
    LocalSandboxProviderConfig,
    SandboxArtifact,
    SandboxConnectionInfo,
    SandboxEnvironmentSpec,
    SandboxExecRequest,
    SandboxExecResult,
    SandboxOperationContext,
    SandboxOperationStatus,
    SandboxReadResult,
    SandboxSessionSpec,
    SandboxTaskSpec,
    SandboxWriteResult,
)

if TYPE_CHECKING:
    from asyncio import StreamReader
    from asyncio.subprocess import Process

    from pyrit.executor.capability.evidence import CapabilityEvidenceSink
    from pyrit.models import JSONValue


class SandboxPathEscapeError(ValueError):
    """A requested path leaves the environment workspace."""


class SandboxSetupError(RuntimeError):
    """Environment setup did not complete successfully."""


class LocalSandboxProvider(SandboxProvider):
    """
    Execute trusted development workloads in isolated temporary workspaces.

    This provider is not a security boundary. Commands run with the PyRIT process
    identity and can access host resources through executable behavior. Workspace
    path checks protect accidental file escape, not malicious code.
    """

    def __init__(
        self,
        *,
        config: LocalSandboxProviderConfig | None = None,
        evidence_sink: CapabilityEvidenceSink | None = None,
    ) -> None:
        """Initialize local provider configuration."""
        super().__init__()
        self._config = config or LocalSandboxProviderConfig()
        self._evidence_sink = evidence_sink
        self._workspace_root: Path | None = None
        self._owns_workspace_root = False
        self._sessions: dict[str, LocalSandboxSession] = {}
        self._sessions_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """The stable provider name."""
        return "local"

    @property
    def is_security_boundary(self) -> bool:
        """Whether this provider isolates untrusted code."""
        return False

    async def _prepare_async(self) -> None:
        started_at = _now()
        if self._config.workspace_root is None:
            root = await asyncio.to_thread(_create_temp_root)
            self._workspace_root = Path(root).resolve()
            self._owns_workspace_root = True
        else:
            self._workspace_root = self._config.workspace_root.resolve()
            await asyncio.to_thread(self._workspace_root.mkdir, parents=True, exist_ok=True)
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                provider=self.name,
                operation="provider_prepare",
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=started_at,
            ),
        )

    async def _prepare_task_async(self, task: SandboxTaskSpec) -> None:
        started_at = _now()
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                provider=self.name,
                operation="task_prepare",
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=started_at,
                metadata={"task_id_sha256": _hash_bytes(task.task_id.encode())},
            ),
        )

    async def _cleanup_task_async(self, task: SandboxTaskSpec) -> None:
        started_at = _now()
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                provider=self.name,
                operation="task_cleanup",
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=started_at,
                metadata={"task_id_sha256": _hash_bytes(task.task_id.encode())},
            ),
        )

    async def _create_session_async(
        self,
        *,
        spec: SandboxSessionSpec,
        evidence_sink: CapabilityEvidenceSink | None,
    ) -> SandboxSession:
        if self._workspace_root is None:
            raise RuntimeError("Local sandbox provider workspace root is not initialized.")
        started_at = _now()
        session_root = self._workspace_root / f"session-{_safe_id(spec.session_id)}"
        try:
            await asyncio.to_thread(session_root.mkdir, parents=False, exist_ok=False)
            session = LocalSandboxSession(
                provider=self,
                spec=spec,
                root=session_root,
                evidence_sink=evidence_sink or self._evidence_sink,
            )
            async with self._sessions_lock:
                self._sessions[spec.session_id] = session
        except BaseException:
            if session_root.exists():
                await asyncio.to_thread(shutil.rmtree, session_root, True)
            raise
        await session.emit_lifecycle_evidence_async(operation="session_create", started_at=started_at)
        return session

    async def _remove_session_async(self, session_id: str) -> None:
        async with self._sessions_lock:
            self._sessions.pop(session_id, None)

    async def _cleanup_async(self) -> None:
        started_at = _now()
        async with self._sessions_lock:
            sessions = tuple(self._sessions.values())
        results = await asyncio.gather(*(session.close_async() for session in sessions), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if self._workspace_root is not None and self._owns_workspace_root and not self._config.retain_workspaces:
            await asyncio.to_thread(shutil.rmtree, self._workspace_root, True)
        outcome = SandboxOperationStatus.FAILED if failures else SandboxOperationStatus.SUCCEEDED
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                provider=self.name,
                operation="provider_cleanup",
                outcome=outcome,
                started_at=started_at,
                error_code="session_cleanup_failed" if failures else None,
                metadata={"session_cleanup_failures": len(failures)},
            ),
        )
        if failures:
            raise RuntimeError(f"{len(failures)} local sandbox session cleanup operation(s) failed.")

    async def _cleanup_orphans_async(self) -> int:
        if self._workspace_root is None:
            return 0
        started_at = _now()
        workspace_root = self._workspace_root
        async with self._sessions_lock:
            active_roots = {session.root for session in self._sessions.values()}
        candidates = await asyncio.to_thread(
            lambda: tuple(path for path in workspace_root.glob("session-*") if path not in active_roots)
        )
        cleaned_count = 0
        if not self._config.retain_workspaces:
            await asyncio.gather(*(asyncio.to_thread(shutil.rmtree, path, True) for path in candidates))
            cleaned_count = len(candidates)
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                provider=self.name,
                operation="orphan_cleanup",
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=started_at,
                metadata={
                    "resources_discovered": len(candidates),
                    "resources_cleaned": cleaned_count,
                },
            ),
        )
        return cleaned_count


class LocalSandboxSession(SandboxSession):
    """A local per-attempt workspace containing named environment directories."""

    def __init__(
        self,
        *,
        provider: LocalSandboxProvider,
        spec: SandboxSessionSpec,
        root: Path,
        evidence_sink: CapabilityEvidenceSink | None,
    ) -> None:
        """Initialize a local session."""
        super().__init__(provider_name=provider.name, spec=spec, evidence_sink=evidence_sink)
        self._provider = provider
        self.root = root
        self._environments = tuple(
            LocalSandboxEnvironment(
                session=self,
                spec=environment_spec,
                root=root / "environments" / environment_spec.name,
                unrestricted=provider._config.allow_unrestricted_host_execution,
                evidence_sink=evidence_sink,
            )
            for environment_spec in spec.environments
        )

    @property
    def environments(self) -> tuple[SandboxEnvironment, ...]:
        """The session environments."""
        return self._environments

    async def _initialize_async(self) -> None:
        started_at = _now()
        for environment in self._environments:
            await environment.initialize_async()
        await self.emit_lifecycle_evidence_async(operation="session_setup", started_at=started_at)

    async def _close_async(self) -> None:
        started_at = _now()
        await asyncio.gather(*(environment.close_async() for environment in self._environments))
        if not self._provider._config.retain_workspaces:
            await asyncio.to_thread(shutil.rmtree, self.root, True)
        await self._provider._remove_session_async(self.session_id)
        await self.emit_lifecycle_evidence_async(operation="session_cleanup", started_at=started_at)

    async def emit_lifecycle_evidence_async(self, *, operation: str, started_at: datetime) -> None:
        """Emit one session lifecycle event."""
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                provider=self._provider_name,
                operation=operation,
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=started_at,
                session_id=self.session_id,
            ),
        )


class LocalSandboxEnvironment(SandboxEnvironment):
    """A path-contained local execution environment."""

    def __init__(
        self,
        *,
        session: LocalSandboxSession,
        spec: SandboxEnvironmentSpec,
        root: Path,
        unrestricted: bool,
        evidence_sink: CapabilityEvidenceSink | None,
    ) -> None:
        """Initialize a local environment."""
        self._session = session
        self._spec = spec
        self._root = root
        self._resolved_root: Path | None = None
        self._unrestricted = unrestricted
        self._evidence_sink = evidence_sink
        self._path_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._closed = False
        self._process_lock = asyncio.Lock()
        self._processes: set[LocalSandboxProcess] = set()

    @property
    def name(self) -> str:
        """The environment name."""
        return self._spec.name

    @property
    def connection_info(self) -> SandboxConnectionInfo:
        """Non-secret local connection metadata."""
        return SandboxConnectionInfo(
            provider="local",
            session_id=self._session.session_id,
            environment_name=self.name,
            transport="local-process",
            metadata={"security_boundary": False, "workspace_isolated": True},
        )

    async def initialize_async(self) -> None:
        """
        Materialize setup files and execute setup scripts.

        Raises:
            SandboxSetupError: If a setup file or command fails.
        """
        started_at = _now()
        await asyncio.to_thread(self._root.mkdir, parents=True, exist_ok=False)
        self._resolved_root = self._root.resolve()
        for setup_file in self._spec.setup_files:
            result = await self.write_file_async(path=setup_file.path, data=setup_file.content)
            if result.status is not SandboxOperationStatus.SUCCEEDED:
                raise SandboxSetupError(f"A setup file failed with status '{result.status.value}'.")
            if setup_file.executable:
                path = self._resolve_path(setup_file.path)
                await asyncio.to_thread(path.chmod, path.stat().st_mode | stat.S_IXUSR)
        for setup_script in self._spec.setup_scripts:
            result = await self.exec_async(request=setup_script.request)
            if result.status is not SandboxOperationStatus.SUCCEEDED:
                raise SandboxSetupError(
                    f"Setup command failed with status '{result.status.value}' and exit code {result.exit_code}."
                )
        evidence = _evidence(
            provider="local",
            operation="environment_setup",
            outcome=SandboxOperationStatus.SUCCEEDED,
            started_at=started_at,
            session_id=self._session.session_id,
            environment_name=self.name,
        )
        await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)

    async def start_process_async(
        self,
        *,
        request: SandboxExecRequest,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxProcess:
        """
        Start a local subprocess in a new process group.

        Returns:
            SandboxProcess: The running process handle.

        Raises:
            ValueError: If an alternate process user was requested.
            RuntimeError: If the environment is closed.
        """
        async with self._operation_lock:
            if self._closed:
                raise RuntimeError("Sandbox environment is closed.")
            return await self._start_process_unlocked_async(
                request=request,
                operation_context=operation_context,
            )

    async def _start_process_unlocked_async(
        self,
        *,
        request: SandboxExecRequest,
        operation_context: SandboxOperationContext | None,
    ) -> SandboxProcess:
        """
        Start and register a process while holding the operation gate.

        Returns:
            SandboxProcess: The running process handle.

        Raises:
            ValueError: If an alternate process user was requested.
        """
        if request.user is not None:
            raise ValueError("LocalSandboxProvider does not support running as an alternate user.")
        cwd = self._resolve_cwd(request.cwd)
        environment = os.environ.copy()
        environment.update(request.environment)
        command = self._command(request)
        if os.name == "nt":
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        else:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        process_handle = LocalSandboxProcess(
            process=process,
            request=request,
            environment=self,
            operation_context=operation_context,
            started_at=_now(),
            windows_job=_WindowsJob.create(process.pid) if os.name == "nt" else None,
        )
        async with self._process_lock:
            self._processes.add(process_handle)
        return process_handle

    async def exec_async(
        self,
        *,
        request: SandboxExecRequest,
        cancellation_event: asyncio.Event | None = None,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxExecResult:
        """
        Execute a bounded local process.

        Returns:
            SandboxExecResult: Buffered output and terminal status.
        """
        started_at = _now()
        try:
            process = await self.start_process_async(request=request, operation_context=operation_context)
        except (FileNotFoundError, PermissionError, OSError, ValueError, RuntimeError) as error:
            status = (
                SandboxOperationStatus.NOT_FOUND
                if isinstance(error, FileNotFoundError)
                else SandboxOperationStatus.PERMISSION_DENIED
                if isinstance(error, PermissionError)
                else SandboxOperationStatus.FAILED
            )
            error_code, error_message = _process_start_error(error)
            evidence = self._operation_evidence(
                operation="exec",
                outcome=status,
                started_at=started_at,
                operation_context=operation_context,
                error_code=error_code,
            )
            await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)
            return SandboxExecResult(
                status=status,
                error_code=error_code,
                error_message=error_message,
                evidence=(evidence,),
            )
        return await process.communicate_async(stdin=request.stdin, cancellation_event=cancellation_event)

    async def read_file_async(
        self,
        *,
        path: str,
        max_bytes: int | None = None,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxReadResult:
        """
        Read a bounded binary file.

        Returns:
            SandboxReadResult: File data or an explicit error status.
        """
        async with self._operation_lock:
            if self._closed:
                started_at = _now()
                return await self._read_error_async(
                    status=SandboxOperationStatus.FAILED,
                    path=path,
                    started_at=started_at,
                    operation_context=operation_context,
                    error_code="environment_closed",
                    error_message="Sandbox environment is closed.",
                )
            return await self._read_file_unlocked_async(
                path=path,
                max_bytes=max_bytes,
                operation_context=operation_context,
            )

    async def _read_file_unlocked_async(
        self,
        *,
        path: str,
        max_bytes: int | None,
        operation_context: SandboxOperationContext | None,
    ) -> SandboxReadResult:
        """
        Read a file while holding the environment operation gate.

        Returns:
            SandboxReadResult: File data or an explicit error status.
        """
        started_at = _now()
        if max_bytes is not None and max_bytes <= 0:
            return await self._read_error_async(
                status=SandboxOperationStatus.FAILED,
                path=path,
                started_at=started_at,
                operation_context=operation_context,
                error_code="invalid_read_limit",
                error_message="Read limit must be greater than zero.",
            )
        limit = min(max_bytes or self._spec.limits.max_read_bytes, self._spec.limits.max_read_bytes)
        try:
            resolved = self._resolve_path(path)
            async with aiofiles.open(resolved, "rb") as file:
                data = await file.read(limit + 1)
            if len(data) > limit:
                return await self._read_error_async(
                    status=SandboxOperationStatus.TOO_LARGE,
                    path=path,
                    started_at=started_at,
                    operation_context=operation_context,
                    error_code="read_limit_exceeded",
                    error_message=f"File exceeds read limit {limit}.",
                    size_bytes=len(data),
                )
        except SandboxPathEscapeError:
            return await self._read_error_async(
                status=SandboxOperationStatus.PATH_ESCAPE,
                path=path,
                started_at=started_at,
                operation_context=operation_context,
                error_code="path_escape",
                error_message="Requested path is outside the sandbox environment.",
            )
        except FileNotFoundError:
            return await self._read_error_async(
                status=SandboxOperationStatus.NOT_FOUND,
                path=path,
                started_at=started_at,
                operation_context=operation_context,
                error_code="file_not_found",
                error_message="File was not found.",
            )
        except PermissionError:
            return await self._read_error_async(
                status=SandboxOperationStatus.PERMISSION_DENIED,
                path=path,
                started_at=started_at,
                operation_context=operation_context,
                error_code="permission_denied",
                error_message="File permission was denied.",
            )
        except OSError:
            return await self._read_error_async(
                status=SandboxOperationStatus.FAILED,
                path=path,
                started_at=started_at,
                operation_context=operation_context,
                error_code="filesystem_error",
                error_message="File read failed.",
            )
        sha256 = _hash_bytes(data)
        artifact = self._file_artifact(path=path, size_bytes=len(data), sha256=sha256)
        evidence = self._operation_evidence(
            operation="read",
            outcome=SandboxOperationStatus.SUCCEEDED,
            started_at=started_at,
            operation_context=operation_context,
            output_size_bytes=len(data),
            sha256=sha256,
            artifact_ids=(artifact.artifact_id,),
        )
        await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)
        return SandboxReadResult(
            status=SandboxOperationStatus.SUCCEEDED,
            data=data,
            size_bytes=len(data),
            sha256=sha256,
            artifacts=(artifact,),
            evidence=(evidence,),
        )

    async def write_file_async(
        self,
        *,
        path: str,
        data: bytes,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxWriteResult:
        """
        Write a bounded binary file.

        Returns:
            SandboxWriteResult: Write metadata or an explicit error status.
        """
        async with self._operation_lock:
            if self._closed:
                return await self._write_error_async(
                    status=SandboxOperationStatus.FAILED,
                    started_at=_now(),
                    operation_context=operation_context,
                    error_code="environment_closed",
                    error_message="Sandbox environment is closed.",
                    input_size_bytes=len(data),
                )
            return await self._write_file_unlocked_async(
                path=path,
                data=data,
                operation_context=operation_context,
            )

    async def _write_file_unlocked_async(
        self,
        *,
        path: str,
        data: bytes,
        operation_context: SandboxOperationContext | None,
    ) -> SandboxWriteResult:
        """
        Write a file while holding the environment operation gate.

        Returns:
            SandboxWriteResult: Write metadata or an explicit error status.
        """
        started_at = _now()
        if len(data) > self._spec.limits.max_write_bytes:
            return await self._write_error_async(
                status=SandboxOperationStatus.TOO_LARGE,
                started_at=started_at,
                operation_context=operation_context,
                error_code="write_limit_exceeded",
                error_message=f"Data size {len(data)} exceeds write limit {self._spec.limits.max_write_bytes}.",
                input_size_bytes=len(data),
            )
        write_started = False
        try:
            async with self._path_lock:
                resolved = self._resolve_path(path)
                await asyncio.to_thread(resolved.parent.mkdir, parents=True, exist_ok=True)
            async with aiofiles.open(resolved, "wb") as file:
                write_started = True
                await file.write(data)
        except SandboxPathEscapeError:
            return await self._write_error_async(
                status=SandboxOperationStatus.PATH_ESCAPE,
                started_at=started_at,
                operation_context=operation_context,
                error_code="path_escape",
                error_message="Requested path is outside the sandbox environment.",
                input_size_bytes=len(data),
            )
        except PermissionError:
            return await self._write_error_async(
                status=SandboxOperationStatus.PERMISSION_DENIED,
                started_at=started_at,
                operation_context=operation_context,
                error_code="permission_denied",
                error_message="File permission was denied.",
                input_size_bytes=len(data),
            )
        except OSError:
            return await self._write_error_async(
                status=SandboxOperationStatus.FAILED,
                started_at=started_at,
                operation_context=operation_context,
                error_code="filesystem_error",
                error_message="File write failed.",
                input_size_bytes=len(data),
                side_effect_completed=None if write_started else False,
            )
        sha256 = _hash_bytes(data)
        artifact = self._file_artifact(path=path, size_bytes=len(data), sha256=sha256)
        evidence = self._operation_evidence(
            operation="write",
            outcome=SandboxOperationStatus.SUCCEEDED,
            started_at=started_at,
            operation_context=operation_context,
            input_size_bytes=len(data),
            sha256=sha256,
            artifact_ids=(artifact.artifact_id,),
        )
        await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)
        return SandboxWriteResult(
            status=SandboxOperationStatus.SUCCEEDED,
            size_bytes=len(data),
            sha256=sha256,
            artifacts=(artifact,),
            evidence=(evidence,),
        )

    def _resolve_path(self, path: str) -> Path:
        root = self._resolved_root or self._root.resolve()
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise SandboxPathEscapeError(f"Path '{path}' leaves sandbox environment '{self.name}'.")
        return candidate

    async def close_async(self) -> None:
        """Terminate every active process before session workspace cleanup."""
        async with self._operation_lock:
            self._closed = True
            async with self._process_lock:
                processes = tuple(self._processes)
        await asyncio.gather(*(process.terminate_async() for process in processes))
        async with self._process_lock:
            self._processes.difference_update(processes)

    async def remove_process_async(self, process: LocalSandboxProcess) -> None:
        """Remove a completed process from environment tracking."""
        async with self._process_lock:
            self._processes.discard(process)

    def _resolve_cwd(self, cwd: str | None) -> Path:
        if cwd is None:
            return self._root
        requested = Path(cwd)
        if requested.is_absolute():
            if not self._unrestricted:
                raise SandboxPathEscapeError("Absolute process cwd requires allow_unrestricted_host_execution=True.")
            return requested
        return self._resolve_path(cwd)

    @staticmethod
    def _command(request: SandboxExecRequest) -> tuple[str, ...]:
        if request.argv is not None:
            return request.argv
        if request.shell_script is None:
            raise ValueError("Shell-script request is missing script content.")
        if os.name == "nt":
            return (os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", request.shell_script)
        return ("/bin/sh", "-c", request.shell_script)

    def _file_artifact(self, *, path: str, size_bytes: int, sha256: str) -> SandboxArtifact:
        artifact_id = f"{self._session.session_id}:{self.name}:{path}"
        return SandboxArtifact(
            artifact_id=artifact_id,
            uri=f"sandbox://{self._session.session_id}/{self.name}/{path}",
            sha256=sha256,
            size_bytes=size_bytes,
        )

    def _operation_evidence(
        self,
        *,
        operation: str,
        outcome: SandboxOperationStatus,
        started_at: datetime,
        operation_context: SandboxOperationContext | None = None,
        error_code: str | None = None,
        input_size_bytes: int | None = None,
        output_size_bytes: int | None = None,
        sha256: str | None = None,
        artifact_ids: tuple[str, ...] = (),
        metadata: dict[str, JSONValue] | None = None,
    ) -> SandboxOperationEvidence:
        return _evidence(
            provider="local",
            operation=operation,
            outcome=outcome,
            started_at=started_at,
            session_id=self._session.session_id,
            environment_name=self.name,
            operation_context=operation_context,
            error_code=error_code,
            input_size_bytes=input_size_bytes,
            output_size_bytes=output_size_bytes,
            sha256=sha256,
            artifact_ids=artifact_ids,
            metadata=metadata,
        )

    async def _read_error_async(
        self,
        *,
        status: SandboxOperationStatus,
        path: str,
        started_at: datetime,
        operation_context: SandboxOperationContext | None,
        error_code: str,
        error_message: str,
        size_bytes: int | None = None,
    ) -> SandboxReadResult:
        evidence = self._operation_evidence(
            operation="read",
            outcome=status,
            started_at=started_at,
            operation_context=operation_context,
            error_code=error_code,
            output_size_bytes=size_bytes,
            metadata={"path_sha256": _hash_bytes(path.encode())},
        )
        await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)
        return SandboxReadResult(
            status=status,
            size_bytes=size_bytes,
            error_code=error_code,
            error_message=error_message,
            evidence=(evidence,),
        )

    async def _write_error_async(
        self,
        *,
        status: SandboxOperationStatus,
        started_at: datetime,
        operation_context: SandboxOperationContext | None,
        error_code: str,
        error_message: str,
        input_size_bytes: int,
        side_effect_completed: bool | None = False,
    ) -> SandboxWriteResult:
        evidence = self._operation_evidence(
            operation="write",
            outcome=status,
            started_at=started_at,
            operation_context=operation_context,
            error_code=error_code,
            input_size_bytes=input_size_bytes,
        )
        await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)
        return SandboxWriteResult(
            status=status,
            error_code=error_code,
            error_message=error_message,
            side_effect_completed=side_effect_completed,
            evidence=(evidence,),
        )


class LocalSandboxProcess(SandboxProcess):
    """A local process with bounded output and process-tree termination."""

    def __init__(
        self,
        *,
        process: Process,
        request: SandboxExecRequest,
        environment: LocalSandboxEnvironment,
        operation_context: SandboxOperationContext | None,
        started_at: datetime,
        windows_job: _WindowsJob | None,
    ) -> None:
        """Initialize the process handle."""
        self._process = process
        self._request = request
        self._environment = environment
        self._operation_context = operation_context
        self._started_at = started_at
        self._windows_job = windows_job
        self._termination_lock = asyncio.Lock()
        self._termination_task: asyncio.Task[None] | None = None

    async def communicate_async(
        self,
        *,
        stdin: bytes | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> SandboxExecResult:
        """
        Wait for completion while bounding output and honoring cancellation.

        Returns:
            SandboxExecResult: Buffered output and terminal status.

        Raises:
            asyncio.CancelledError: If the calling task is cancelled.
        """
        timeout = min(
            self._request.timeout_seconds or self._environment._spec.limits.max_exec_seconds,
            self._environment._spec.limits.max_exec_seconds,
        )
        deadline = asyncio.get_running_loop().time() + timeout
        stdout_task = asyncio.create_task(
            _read_limited_async(
                stream=self._process.stdout,
                limit=self._environment._spec.limits.max_stdout_bytes,
            )
        )
        stderr_task = asyncio.create_task(
            _read_limited_async(
                stream=self._process.stderr,
                limit=self._environment._spec.limits.max_stderr_bytes,
            )
        )
        stdin_task = asyncio.create_task(self._write_stdin_async(stdin))
        wait_task = asyncio.create_task(self._process.wait())
        cancellation_task = asyncio.create_task(cancellation_event.wait()) if cancellation_event is not None else None
        timed_out = False
        cancelled = False
        stdout: tuple[bytes, bool] = (b"", False)
        stderr: tuple[bytes, bool] = (b"", False)
        try:
            waiting = (wait_task,) if cancellation_task is None else (wait_task, cancellation_task)
            done, _ = await asyncio.wait(
                waiting,
                timeout=_remaining_seconds(deadline),
                return_when=asyncio.FIRST_COMPLETED,
            )
            timed_out = not done
            cancelled = cancellation_task is not None and cancellation_task in done
            if not timed_out and not cancelled:
                try:
                    stdout, stderr = await asyncio.wait_for(
                        asyncio.gather(stdout_task, stderr_task),
                        timeout=_remaining_seconds(deadline),
                    )
                    await asyncio.wait_for(stdin_task, timeout=_remaining_seconds(deadline))
                except asyncio.TimeoutError:
                    timed_out = True
            if timed_out or cancelled:
                await self.terminate_async()
                stdin_task.cancel()
                stdout, stderr = await self._finish_streams_async(
                    stdout_task=stdout_task,
                    stderr_task=stderr_task,
                )
        except asyncio.CancelledError:
            await asyncio.shield(self.terminate_async())
            evidence = self._environment._operation_evidence(
                operation="exec",
                outcome=SandboxOperationStatus.CANCELLED,
                started_at=self._started_at,
                operation_context=self._operation_context,
                error_code="task_cancelled",
            )
            await asyncio.shield(_emit_evidence_async(sink=self._environment._evidence_sink, evidence=evidence))
            raise
        except BaseException:
            await asyncio.shield(self.terminate_async())
            raise
        finally:
            if cancellation_task is not None:
                cancellation_task.cancel()
            for task in (stdin_task, wait_task, stdout_task, stderr_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stdin_task, wait_task, stdout_task, stderr_task, return_exceptions=True)
            self._close_windows_job()
            await self._environment.remove_process_async(self)
        await self.terminate_async()
        stdout_bytes, stdout_truncated = stdout
        stderr_bytes, stderr_truncated = stderr
        return await self._build_result_async(
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            timed_out=timed_out,
            cancelled=cancelled,
        )

    async def terminate_async(self) -> None:
        """Terminate the process group or tree exactly once."""
        async with self._termination_lock:
            if self._termination_task is None:
                self._termination_task = asyncio.create_task(self._terminate_tree_async())
            termination_task = self._termination_task
        await asyncio.shield(termination_task)

    async def _terminate_tree_async(self) -> None:
        """Terminate the owned process tree and await the direct process."""
        if os.name == "nt":
            await self._terminate_windows_tree_async()
        else:
            await self._terminate_posix_group_async()
        await self._process.wait()

    async def _terminate_posix_group_async(self) -> None:
        try:
            os.kill(-self._process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = asyncio.get_running_loop().time() + self._environment._spec.limits.terminate_grace_seconds
        while _remaining_seconds(deadline) > 0:
            try:
                os.kill(-self._process.pid, 0)
            except ProcessLookupError:
                return
            await asyncio.sleep(min(0.05, _remaining_seconds(deadline)))
        try:
            os.kill(-self._process.pid, signal.Signals(9))
        except ProcessLookupError:
            return

    async def _terminate_windows_tree_async(self) -> None:
        if self._windows_job is not None:
            await asyncio.to_thread(self._windows_job.close)
            return
        try:
            terminator = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(self._process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await terminator.wait()
        except FileNotFoundError:
            if self._process.returncode is None:
                self._process.kill()

    async def _write_stdin_async(self, data: bytes | None) -> None:
        if self._process.stdin is None:
            return
        if data:
            self._process.stdin.write(data)
            with suppress(BrokenPipeError, ConnectionResetError):
                await self._process.stdin.drain()
        self._process.stdin.close()

    async def _finish_streams_async(
        self,
        *,
        stdout_task: asyncio.Task[tuple[bytes, bool]],
        stderr_task: asyncio.Task[tuple[bytes, bool]],
    ) -> tuple[tuple[bytes, bool], tuple[bytes, bool]]:
        grace = max(1.0, self._environment._spec.limits.terminate_grace_seconds)
        try:
            return await asyncio.wait_for(asyncio.gather(stdout_task, stderr_task), timeout=grace)
        except asyncio.TimeoutError:
            stdout_task.cancel()
            stderr_task.cancel()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            return (b"", True), (b"", True)

    def _close_windows_job(self) -> None:
        if self._windows_job is not None:
            self._windows_job.close()

    async def _build_result_async(
        self,
        *,
        stdout: bytes,
        stderr: bytes,
        stdout_truncated: bool,
        stderr_truncated: bool,
        timed_out: bool,
        cancelled: bool,
    ) -> SandboxExecResult:
        if cancelled:
            status = SandboxOperationStatus.CANCELLED
            error_code = "cancelled"
        elif timed_out:
            status = SandboxOperationStatus.TIMED_OUT
            error_code = "timeout"
        elif stdout_truncated or stderr_truncated:
            status = SandboxOperationStatus.TRUNCATED
            error_code = "output_limit_exceeded"
        else:
            status = (
                SandboxOperationStatus.SUCCEEDED if self._process.returncode == 0 else SandboxOperationStatus.FAILED
            )
            error_code = None if status is SandboxOperationStatus.SUCCEEDED else "nonzero_exit"
        evidence = self._environment._operation_evidence(
            operation="exec",
            outcome=status,
            started_at=self._started_at,
            operation_context=self._operation_context,
            error_code=error_code,
            input_size_bytes=len(self._request.stdin or b""),
            output_size_bytes=len(stdout) + len(stderr),
            metadata={
                "exit_code": self._process.returncode,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "command_form": "argv" if self._request.argv is not None else "shell_script",
            },
        )
        await _emit_evidence_async(sink=self._environment._evidence_sink, evidence=evidence)
        return SandboxExecResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=self._process.returncode,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            timed_out=timed_out,
            cancelled=cancelled,
            error_code=error_code,
            evidence=(evidence,),
        )


async def _read_limited_async(*, stream: StreamReader | None, limit: int) -> tuple[bytes, bool]:
    """
    Drain a process stream while retaining at most the configured byte limit.

    Returns:
        tuple[bytes, bool]: Retained bytes and whether output was truncated.
    """
    if stream is None:
        return b"", False
    retained = bytearray()
    truncated = False
    while chunk := await stream.read(65_536):
        remaining = limit - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
    return bytes(retained), truncated


class _JobBasicLimitInformation(ctypes.Structure):
    """Windows Job Object basic limits."""

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    """Windows Job Object I/O counters."""

    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobExtendedLimitInformation(ctypes.Structure):
    """Windows Job Object extended limits."""

    _fields_ = [
        ("BasicLimitInformation", _JobBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _WindowsJob:
    """A Windows Job Object that terminates all assigned descendants on close."""

    KILL_ON_JOB_CLOSE = 0x00002000
    EXTENDED_LIMIT_INFORMATION_CLASS = 9
    PROCESS_TERMINATE = 0x0001
    PROCESS_SET_QUOTA = 0x0100

    def __init__(self, *, kernel32: Any, handle: Any) -> None:
        """Initialize an owned Job Object handle."""
        self._kernel32 = kernel32
        self._handle = handle

    @classmethod
    def create(cls, process_id: int) -> _WindowsJob | None:
        """
        Create a kill-on-close job and assign a process.

        Returns:
            _WindowsJob | None: The assigned job, or None when the host disallows assignment.
        """
        kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.OpenProcess.restype = wintypes.HANDLE
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        information = _JobExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = cls.KILL_ON_JOB_CLOSE
        configured = kernel32.SetInformationJobObject(
            job,
            cls.EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(information),
            ctypes.sizeof(information),
        )
        process = kernel32.OpenProcess(cls.PROCESS_TERMINATE | cls.PROCESS_SET_QUOTA, False, process_id)
        assigned = process and kernel32.AssignProcessToJobObject(job, process)
        if process:
            kernel32.CloseHandle(process)
        if not configured or not assigned:
            kernel32.CloseHandle(job)
            return None
        return cls(kernel32=kernel32, handle=job)

    def close(self) -> None:
        """Close the job once, terminating every assigned process."""
        handle = self._handle
        if not handle:
            return
        self._handle = None
        self._kernel32.CloseHandle(handle)


def _evidence(
    *,
    provider: str,
    operation: str,
    outcome: SandboxOperationStatus,
    started_at: datetime,
    session_id: str | None = None,
    environment_name: str | None = None,
    operation_context: SandboxOperationContext | None = None,
    error_code: str | None = None,
    input_size_bytes: int | None = None,
    output_size_bytes: int | None = None,
    sha256: str | None = None,
    artifact_ids: tuple[str, ...] = (),
    metadata: dict[str, JSONValue] | None = None,
) -> SandboxOperationEvidence:
    """
    Build a content-free authoritative sandbox evidence event.

    Returns:
        SandboxOperationEvidence: The immutable evidence event.
    """
    return SandboxOperationEvidence(
        provider=provider,
        operation=operation,
        outcome=outcome.value,
        started_at=started_at,
        ended_at=_now(),
        session_id=session_id,
        environment_name=environment_name,
        call_id=operation_context.call_id if operation_context else None,
        attempt_id=operation_context.attempt_id if operation_context else None,
        error_code=error_code,
        input_size_bytes=input_size_bytes,
        output_size_bytes=output_size_bytes,
        sha256=sha256,
        artifact_ids=artifact_ids,
        metadata=metadata or {},
    )


async def _emit_evidence_async(
    *,
    sink: CapabilityEvidenceSink | None,
    evidence: SandboxOperationEvidence,
) -> None:
    """Emit evidence when a sink was configured."""
    if sink is not None:
        await sink.emit_async(evidence)


def _hash_bytes(data: bytes) -> str:
    """Return a SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()


def _safe_id(value: str) -> str:
    """Return a deterministic filesystem-safe identifier."""
    return _hash_bytes(value.encode())[:20]


def _process_start_error(error: OSError | ValueError | RuntimeError) -> tuple[str, str]:
    """
    Map process startup failures to stable content-free errors.

    Returns:
        tuple[str, str]: Error code and sanitized message.
    """
    if isinstance(error, FileNotFoundError):
        return "executable_not_found", "Executable was not found."
    if isinstance(error, PermissionError):
        return "permission_denied", "Executable permission was denied."
    if isinstance(error, SandboxPathEscapeError):
        return "path_escape", "Process working directory is outside the sandbox environment."
    if isinstance(error, RuntimeError):
        return "environment_closed", "Sandbox environment is closed."
    if isinstance(error, ValueError):
        return "unsupported_process_option", "The requested process option is not supported."
    return "process_start_failed", "The process could not be started."


def _remaining_seconds(deadline: float) -> float:
    """
    Return non-negative time remaining before a monotonic deadline.

    Returns:
        float: Remaining seconds.
    """
    return max(0.0, deadline - asyncio.get_running_loop().time())


def _create_temp_root() -> str:
    """
    Create the provider temporary workspace root.

    Returns:
        str: The new temporary directory path.
    """
    return tempfile.mkdtemp(prefix="pyrit-sandbox-")


def _now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(tz=timezone.utc)
